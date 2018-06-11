from multiprocessing import cpu_count
import sys

print('CPUs: ' + str(cpu_count()))
sys.stdout.flush()
