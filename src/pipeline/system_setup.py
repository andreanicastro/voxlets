# class to set up the parameters for the system...
import socket

host_name = socket.gethostname()

# note: fewer testing cores as it tends to be more memory-intensive...
if True:
    # host_name == 'troll' or host_name == 'biryani':
    small_sample = False
    # print "WARN- should be 500" *10
    max_sequences = 500
    max_test_sequences = 500
    cores = 4
    testing_cores = 6
    multicore = 0
else:
    small_sample = True
    max_sequences = 4
    cores = 8
    testing_cores = 8
    multicore = True
