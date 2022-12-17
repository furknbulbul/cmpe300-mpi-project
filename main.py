from mpi4py import MPI
import sys

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

Nworkers = size-1

# read args with flags
def read_args():
    args =sys.argv
    for i in range(len(args)):
        if(args[i]=="--input_file"):
            input_file = args[i+1] 
        elif (args[i]=="--merge_method"):
            merge_method = args[i+1] 
        elif (args[i]=="--test_file"):
            test_file = args[i+1] 
    
    return [input_file, merge_method, test_file]


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
        received_unigrams=comm.recv(source=i, tag=21) # get unigrams from workers with tag 21
        received_bigrams=comm.recv(source=i, tag=22)  # get unigrams from workers with tag 22

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
    
    #send workers to line count that they will process
    for i in range(Nworkers):
        comm.send(data_counts[i], dest=i+1, tag=1)

# run when merge method is MASTER
def evenly_distributed_method(input_file, test_file):
    
    # master process
    if rank == 0:
        
        #read files
        sample_file = open(input_file, 'r')
        lines = sample_file.readlines()

        test_file = open(test_file, 'r')
        test_lines = test_file.readlines()
        
        distribute_lines(len(lines)) # send line counts to the workers
        
        for i in range(len(lines)):
            comm.send(lines[i], dest=i%Nworkers+1, tag=i) # send lines one by one to the workers
        
        # accumulated data
        acc_bigram ={}
        acc_unigram ={}
        master_merge(acc_bigram, acc_unigram) # merge data 
        evaluate_test_data(test_lines, acc_bigram, acc_unigram) # evaluate probabilities

    # worker processes 
    else:
        line_count = comm.recv(source=0, tag=1) # each workers get line count information  
        print("received number of sentences is {} by worker {}".format(line_count, rank) )

        # holds data for every worker 
        unigrams={}
        bigrams={}

        for i in range(line_count):
            
            # receive line from master
            line = comm.recv(source=0, tag=Nworkers*i+rank-1)
            
            #parse line
            tokens = line.split()
            sentence_length = len(tokens)-1 # do not count </s>
            
            # put tokens into dictionary, or increment value if exist 
            for j in range(1, sentence_length):
                unigrams[tokens[j]] = unigrams.get(tokens[j], 0) + 1
            
            # put tuple of tokens into dictionary, or increment value if exist
            for j in range(1, sentence_length-1):
                bigrams[(tokens[j], tokens[j+1])] = bigrams.get((tokens[j], tokens[j+1]), 0) + 1

        # send calculated data to master with tags 21 and 22 
        comm.send(unigrams, dest = 0, tag=21)
        comm.send(bigrams, dest = 0, tag=22)
        

# run when merge method is WORKERS
def sequential_method(input_file, test_file):
    
    # master process
    if rank == 0:
        
        #read files
        sample_file = open(input_file, 'r')
        lines = sample_file.readlines()

        test_file = open(test_file, 'r')
        test_lines = test_file.readlines()
        
        distribute_lines(len(lines)) # send line counts to the workers
        
        for i in range(len(lines)):
            comm.send(lines[i], dest=i%Nworkers+1, tag=i) # send lines one by one to the workers
        
        # accumulated data
        received_unigrams = comm.recv(source = Nworkers, tag=1)
        received_bigrams = comm.recv(source = Nworkers, tag=2)
        
        evaluate_test_data(test_lines, received_bigrams, received_unigrams) # evaluate probabilities

    #worker processes
    else:
        line_count = comm.recv(source=0, tag=1) # each workers get lines count information  
        print("received number of sentences is {} by worker {}".format(line_count, rank) )

        # holds data for every worker 
        unigrams={}
        bigrams={}

        for i in range(line_count):
            
            #receive line from master
            line = comm.recv(source=0, tag=Nworkers*i+rank-1)

            #parse line
            tokens = line.split()
            sentence_length = len(tokens)-1 # do not count </s>
            
            # put tokens into dictionary, or increment value if exist 
            for j in range(1, sentence_length):
                unigrams[tokens[j]] = unigrams.get(tokens[j], 0) + 1
            
            # put tuple of tokens into dictionary, or increment value if exist
            for j in range(1, sentence_length-1):
                bigrams[(tokens[j], tokens[j+1])] = bigrams.get((tokens[j], tokens[j+1]), 0) + 1

        # get data from previous worker and merge with its calculated data
        if(rank > 1):
            received_unigrams = comm.recv(source = rank-1, tag = 1) # receive unigram data from previous worker
            received_bigrams = comm.recv(source = rank-1, tag = 2) # receive bigram data form previous worker

            for unigram in received_unigrams.items():
                word = unigram[0]
                freq = unigram[1]
                unigrams[word] = unigrams.get(word, 0) + freq

            for bigram in received_bigrams.items():
                couple_word = bigram[0]
                freq = bigram[1]
                bigrams[couple_word] = bigrams.get(couple_word, 0) + freq
    
        if(rank != Nworkers):
            dest = rank + 1
        else:
            dest = 0
        
        # send data to next worker
        comm.send(unigrams, dest = dest, tag = 1)
        comm.send(bigrams, dest = dest, tag = 2)


args = read_args()
input_file = args[0]
merge_method = args[1]
test_file = args[2]

if (merge_method == "MASTER"):
    evenly_distributed_method(input_file, test_file)
else:
    sequential_method(input_file, test_file)
