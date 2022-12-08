from mpi4py import MPI
import sys

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

Nworkers = size-1


#evenly send the numbers of lines to the workers
def distribute_lines(line_count):
    data_counts = [line_count//Nworkers for i in range(Nworkers)]

    for i in range(line_count%Nworkers):
        data_counts[i]+=1
    
    for i in range(Nworkers):
        comm.send(data_counts[i], dest=i+1, tag=1)

#master process
if rank == 0:
    
    sample_name = sys.argv[1]
    merge_method = sys.argv[2]
    test_name = sys.argv[3]

    sample_file = open(sample_name, 'r')
    lines = sample_file.readlines()

    test_file = open(test_name, 'r')
    test_lines = test_file.readlines()
    
    distribute_lines(len(lines))
    
    
    for i in range(len(lines)):
        comm.send(lines[i], dest=i%Nworkers+1, tag=i) #send lines one by one to the workers
    
    if (merge_method == "MASTER"):
        acc_unigram = {}
        acc_bigram = {}
        
        for i in range(1, size):
            received_unigrams=comm.recv(source=i, tag=21)
            received_bigrams=comm.recv(source=i, tag=22)

            for unigram in received_unigrams.items():
                word = unigram[0]
                freq = unigram[1]
                acc_unigram[word] = acc_unigram.get(word, 0) + freq
            
            for bigram in received_bigrams.items():
                couple_word = bigram[0]
                freq = bigram[1]
                acc_bigram[couple_word] = acc_bigram.get(couple_word, 0) + freq

    
    for test in test_lines:
        test = test.strip()
        test_unigram=test.split()[0]
        test_bigram=(test.split()[0], test.split()[1])
        try:
            probability=acc_bigram[test_bigram]/acc_unigram[test_unigram]
        except: 
            probability = 0
        print("probability of bigram \"{}\" is {}".format(test, probability))

else:
    print("worker rank:"+ str(rank))
    data_count = comm.recv(source=0, tag=1)
    print("received number of sentences: "+ str(data_count))

    unigrams={}
    bigrams={}
    for i in range(data_count):
        
        line = comm.recv(source=0, tag=Nworkers*i+rank-1)
        #print("line received by worker {}: {}".format(rank, line))
        tokens = line.split()
        sentence_length = len(tokens)-1
        
        for j in range(1, sentence_length):
            unigrams[tokens[j]] = unigrams.get(tokens[j], 0) + 1
        for j in range(1, sentence_length-1):
            bigrams[(tokens[j], tokens[j+1])] = bigrams.get((tokens[j], tokens[j+1]), 0) + 1

        #print("unigram sent to the master by {}: {}".format(rank, unigrams))
        #print("bigram sent to the master by {}: {}".format(rank, bigrams))

    comm.send(unigrams, dest = 0, tag=21)
    comm.send(bigrams, dest = 0, tag=22)
    




    
   
   

