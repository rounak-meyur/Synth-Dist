#!/usb/bin/env nextflow
params.container = "synthdist:1.0"


params.data_directory = 'data'
params.outdir = 'cameo_output'
params.util_directory = 'utils'
params.model_directory = 'models'
params.config_directory = 'configs'
params.script = 'main_secnet.py'


//component 1: secondary network generation 
process secnet {
    // Use a tag to help identify different process executions
    tag "${x}"

    container = params.container
    label 'parallel'

    // Each parallel execution will publish to a uniquely named subdirectory
    publishDir "$params.outdir", 
        mode: 'copy',
        saveAs: { filename ->
            filename.replace('out', "${x}_out")
        }

    input:
    path script
    path configs_dir
    path data_dir
    path utils_dir
    path models_dir
    each x

    output:
    path "out"
    
    script:
    """
    python ${script} -r ${x}
    """
}



workflow {
    
    
    main_script_ch = Channel.fromPath(params.script, checkIfExists: true)
    config_directory_ch = Channel.fromPath(params.config_directory, checkIfExists: true)
    data_directory_ch = Channel.fromPath(params.data_directory, checkIfExists: true)
    util_directory_ch = Channel.fromPath(params.util_directory, checkIfExists: true)
    model_directory_ch = Channel.fromPath(params.model_directory, checkIfExists: true)

    // List of counties
    region_ch = Channel.of('001','003','005','007','009','011','013','015','017','019','021','023','025','027','029','031','033','035','036','037','041','043','045','047','049','051','053','057','059','061','063','065','067','069','071').flatten()
    
    // Stage: Generate secondary networks
    out_dir = secnet(main_script_ch, config_directory_ch, data_directory_ch, util_directory_ch, model_directory_ch, region_ch)

}