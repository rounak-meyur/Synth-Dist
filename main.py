import warnings
warnings.filterwarnings("ignore")

from tests.SynthDistFixture import SynthDist

fx = SynthDist()
fx.out_dir = "./out/test"
fx.fig_dir = "./figs/test"
fx.grb_dir = "./gurobi/test"
fx.fis = "test"

homes,roads = fx.read_inputs()

map_h2r = fx.map_inputs(
    homes=homes, 
    roads=roads,
    to_file = f"{fx.fis}-map_h2r.txt"
    )


map_r2h = fx.get_reverse_map(
    map_file = "test-map_h2r.txt",
    to_file = "test-map_r2h.txt"
    )

fx.get_secnet_region(map_r2h, homes, roads)