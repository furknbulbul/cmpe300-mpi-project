from mpi4py import MPI
import sys

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

Nworkers = size-1


# calculate probability by using accumulated bigrams and unigrams
def evaluate_test_data(test_lines, acc_bigram, acc_unigram):
    for test in test_lines:
        # parse test words
        test = test.strip()
        test_unigram=test.split()[0]
        test_bigram=(test.split()[0], test.split()[1])

        try:
            probability=acc_bigram[test_bigram]/acc_unigram[test_unigram] # prob = count of bigram[two words]/ count of unigram[first word] 
        except: 
            probability = 0
    
        print("probability of bigram \"{}\" is {}".format(test, probability))



# master merges all data comes from workers
def master_merge(acc_bigram, acc_unigram):
    
    for i in range(1, size):
        received_unigrams=comm.recv(source=i, tag=21) # get unigrams from workers with tag 21(random tag)
        received_bigrams=comm.recv(source=i, tag=22)  # get unigrams from workers with tag 22(random tag)

        # accumulate unigrams
        for unigram in received_unigrams.items():
            word = unigram[0]
            freq = unigram[1]
            acc_unigram[word] = acc_unigram.get(word, 0) + freq
        # accumulate bigrams
        for bigram in received_bigrams.items():
            couple_word = bigram[0]
            freq = bigram[1]
            acc_bigram[couple_word] = acc_bigram.get(couple_word, 0) + freq


# calculate the number of lines for the workers
def distribute_lines(line_count):
    data_counts = [line_count//Nworkers for i in range(Nworkers)]

    for i in range(line_count%Nworkers):
        data_counts[i]+=1
    
    for i in range(Nworkers):
        comm.send(data_counts[i], dest=i+1, tag=1)

# master process
if rank == 0:
    
    # args: sample_name, merge_method, test_name
    sample_name = sys.argv[1] 
    merge_method = sys.argv[2]
    test_name = sys.argv[3]

    sample_file = open(sample_name, 'r')
    lines = sample_file.readlines()

    test_file = open(test_name, 'r')
    test_lines = test_file.readlines()
    
    distribute_lines(len(lines)) # arrange line counts
    
    for i in range(len(lines)):
        comm.send(lines[i], dest=i%Nworkers+1, tag=i) # send lines one by one to the workers
    
    # accumulated data
    acc_bigram ={}
    acc_unigram ={}
    if (merge_method == "MASTER"):
        master_merge(acc_bigram, acc_unigram) # merge data 
    
    evaluate_test_data(test_lines, acc_bigram, acc_unigram) # evaluate probabilities

else:
    print("worker rank:"+ str(rank))
    line_count = comm.recv(source=0, tag=1) # each workers get lines count information  
    print("received number of sentences is {} by worker {}".format(line_count, rank) )

    # holds data for every worker 
    unigrams={}
    bigrams={}

    for i in range(line_count):
        
        line = comm.recv(source=0, tag=Nworkers*i+rank-1)
        #print("line received by worker {}: {}".format(rank, line))
        tokens = line.split()
        sentence_length = len(tokens)-1 # do not count </s>
        
        # put tokens into dictionary, or increment value if exist 
        for j in range(1, sentence_length):
            unigrams[tokens[j]] = unigrams.get(tokens[j], 0) + 1
        
        # put tuple of tokens into dictionary, or increment value if exist
        for j in range(1, sentence_length-1):
            bigrams[(tokens[j], tokens[j+1])] = bigrams.get((tokens[j], tokens[j+1]), 0) + 1

        #print("unigram sent to the master by {}: {}".format(rank, unigrams))
        #print("bigram sent to the master by {}: {}".format(rank, bigrams))

    # send calculated data to master with tags 21 and 22 
    comm.send(unigrams, dest = 0, tag=21)
    comm.send(bigrams, dest = 0, tag=22)
    




    
   
   

