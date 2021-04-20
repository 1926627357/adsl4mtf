import matplotlib.pyplot as plt
import numpy as np
import os
import csv


fig, [ax1, ax2, ax3] = plt.subplots(3, 2, sharex=True)

gpu_usage = []
mem_IO = []
mem_usage = []

with open('/home/haiqwa/document/adsl4mtf/bert/auto_parallel/gpu0_5.csv') as fp:
    reader = csv.reader(fp)
    next(reader)
    for row in reader:
        gpu_usage.append(int(row[1][:-2]))
        mem_IO.append(int(row[2][:-2]))
        mem_usage.append(int(row[3][:-3]))
x = range(len(gpu_usage))


ax1[0].plot(x, gpu_usage)
ax1[0].set(#xlabel='gpu usage', 
        ylabel='gpu usage(%)',
       title='AUTO PARALLEL'
       )
ax2[0].plot(x, mem_IO)
ax2[0].set(#xlabel='gpu mem I/O', 
        ylabel='gpu mem I/O(%)',
       #title='GPU MEM I/O'
       )
ax3[0].plot(x, mem_usage)
ax3[0].set(#xlabel='GPU memory usage', 
        ylabel='GPU memory usage(MiB)',
       #title='GPU Memory Utilization'
       )



gpu_usage = []
mem_IO = []
mem_usage = []

with open('/home/haiqwa/document/adsl4mtf/bert/data_parallel/gpu0_5.csv') as fp:
    reader = csv.reader(fp)
    next(reader)
    for row in reader:
        gpu_usage.append(int(row[1][:-2]))
        mem_IO.append(int(row[2][:-2]))
        mem_usage.append(int(row[3][:-3]))
x = range(len(gpu_usage))


ax1[1].plot(x, gpu_usage)
ax1[1].set(#xlabel='gpu usage', 
        # ylabel='gpu usage(%)',
       title='DATA PARALLEL'
       )
ax2[1].plot(x, mem_IO)
# ax2[1].set(#xlabel='gpu mem I/O', 
#         ylabel='gpu mem I/O(%)',
#        #title='GPU MEM I/O'
#        )
ax3[1].plot(x, mem_usage)
# ax3[1].set(#xlabel='GPU memory usage', 
#         ylabel='GPU memory usage(MiB)',
#        #title='GPU Memory Utilization'
#        )
fig.savefig('test.pdf')