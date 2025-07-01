# Run the workflow on Deception with Nextflow
Cameo can be run in two different modes on Deception: single-node and multi-node.  

## Single-node
The single-node method utilizes an sbatch file to execute Nexflow on a single node of Deception, efficiently utilizing the availble cores on a single node. This mode is most suitable for processes that are quick and do not require multiple cores. SLURM settings for these processes can be configured in the launch_nf.sb script.

### Deception single-node profile from nextflow.config

```
  /* Run all processes on a single node similar to running locally. 
  Using the local executor, launch the slurm sbatch file launch_nf.sb on a single node, 
  the sbatch file will launch the tasks locally on that node. */
  deception_single_node {
    process.executor = 'local' // The user creates the sbatch file
    scratch = "/scratch/$USER/"
    params.maxForks = 20 // limit number of parallel processes run on a single node
    apptainer {
        autoMounts = true //automount system and Nextflow directories
        enabled = true // enable apptainer in Nextflow
        // --userns - allow a normal user to act as root inside the container
        // --fakeroot - imporsonate a root user
        // --env MPLCONFIGDIR - set matplotlib directory to a the writable Nextflow task directory
        runOptions = '--userns --fakeroot --env "MPLCONFIGDIR=$NXF_TASK_WORKDIR"'
        cacheDir = "/scratch/$USER/APPTAINER"
    }
  }
```

### SLURM batch file launch_nf.sb
In single-node mode the SLURM settings are controlled in the sbatch file.  The parition, account, number of nodes, number of tasks per node, can all be modified through the sbatch file.  

```shell
#!/bin/bash
#SBATCH --partition slurm
#SBATCH -c 1
#SBATCH --time 12:00:00
#SBATCH --account cameo
#SBATCH --job-name cameo_dsn # job name
#SBATCH -o cameo_deception_single_node_%j.out
#SBATCH -e cameo_deception_single_node_%j.error
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 60

CONFIG=$1
PIPELINE=$2

mkdir -p $APPTAINER_TMPDIR
mkdir -p $APPTAINER_CACHEDIR

# load required modules
module load apptainer
module load python/3.11.5

export NFX_OPTS="-Xms=512m -Xmx=8g"

# activate the python environment
source ~/cameov2/venv/bin/activate

command="/qfs/projects/cameo/bin/nextflow -C ${CONFIG} run ${PIPELINE} -profile deception_single_node --bg"

echo "$command"
$command
```


#### Run the workflow: single-node
```shell
sbatch ~/cameo/launch_nf.sb ~/cameo/nextflow.config ~/cameo/v2_ecomp.nf
```

## Multi-node

The multi-node method leverages Nextflow to distribute the workflow across multiple nodes in the cluster, with each node handling a separate process. This mode is ideal for CPU-intensive processes. SLURM settings for these processes can be adjusted in the nextflow.config file.

### Deception multi-node profile from nextflow.config

```
// Nextflow will create an sbatch file for each task and launch each task on a separate node. 
  deception {
    scratch = "/scratch/$USER/"
    params.container = '/qfs/projects/cameo/apptainer/cameov2.sif'
    apptainer {
        autoMounts = true
        enabled = true
        runOptions = '--userns --fakeroot --env "MPLCONFIGDIR=$NXF_TASK_WORKDIR" --env "HOSTNAME=$HOSTNAME"'
        cacheDir = "/scratch/$USER/APPTAINER"
    }

    process {
        executor = 'slurm' // Nextflow creates an sbatch file for each task
        queue = 'slurm' // Slurm queue [Slurm queue](https://confluence.pnnl.gov/confluence/display/RCWIKI/Schedule+Jobs)
        time = '30 min' // Time limit
        // --cpus-per-task - number of cpus per task
        // --nodes - number of nodes
        // --ntasks - number of tasks per node.  
        // -A - project name
        clusterOptions = "--cpus-per-task 2 --nodes 1 --ntasks 2 -A cameo"

        // Define a second process type for longer running jobs.  Use the same slurm queue but allow the jobs to run for 3 hours. 
        withLabel: 'parallel' {
          queue = 'slurm' // Slurm queue [Slurm queue](https://confluence.pnnl.gov/confluence/display/RCWIKI/Schedule+Jobs)
          time = '3 hour' // Time limit
          // --cpus-per-task - number of cpus per task
          // --nodes - number of nodes
          // --ntasks - number of tasks per node.  
          // -A - project name
          clusterOptions = "--cpus-per-task 3 --nodes 1 --ntasks 16 -A cameo"
       }
    }
  }
```

#### Run the workflow: multi-node
Schedule the Nextflow process on a cluster node through srun. Nextflow will generate and manage the sbatch configurations to run the processes across the cluster.

srun options for the Nextflow process:
-  -A - project name
-  -p - partition or queue
-  --time - time limit
-  -I - exit if resources are not available within the time period specified (seconds)
-  --pty - execute with a pseudo terminal
-  --ntasks-per-node - number of nodes for the Nextflow control process
-  -u - run unbuffered
-  /qfs/projects/cameo/bin/nextflow - run Nextflow with these options
   - v2_ecomp.nf - pipeline script
   - -profile deception - use the deception profile from nextflow.config


#### Run the workflow: multi-node

```shell
srun -A cameo -p slurm --time=600 -I1200 -N 1 --pty --ntasks-per-node=1 -u /qfs/projects/cameo/bin/nextflow run_synth.nf -profile deception
```
