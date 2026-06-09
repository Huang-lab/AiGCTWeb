import aigct
from aigct.container import VEBenchmarkContainer

print(aigct.__file__)

qm = VEBenchmarkContainer("aigct.yaml").query_mgr

df = qm.get_all_variant_effect_source()
print(len(df))
