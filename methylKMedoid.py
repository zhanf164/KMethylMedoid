import argparse
from math import isnan
import os
import sys
import numpy as np
import itertools
from Bio import SeqIO
from skbio import DistanceMatrix
from skbio.sequence import Sequence
from skbio.sequence.distance import hamming
from sklearn.metrics.pairwise import pairwise_distances

def GatherInput(infile):
    '''Reads in the bait fasta file and converts them to two lists- one containing headers and the other containing sequences'''
    headers = []
    seqs = []
    with open(infile) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            headers.append(str(record.id))
            seqs.append(str(record.seq))

    return headers, seqs

def ReverseComplementSequence(seq):
    comps = {"A":"T", "a":"t","T":"A", "t":"a", "G":"C", "g":"c", "C":"G", "c":"g"}
    complement = ""
    for i in seq:
        try:
            complement += comps[i]
        except KeyError:
            print("Seqeunces must only contain A,T,C,G,a,t,c,g.")
            print("Offending sequence: {}".format(seq))
    return complement[::-1]


def FindConversionIndexes(seq, plant=False):
    '''for plants we need to take into account multiple conversion schemes'''

    positions = []

    #loop through every two nucleotides to look for CpG positions
    for i, nuc in enumerate(seq[:-1]):
        if nuc == "C" or nuc == "c":
            if seq[i+1] == "G" or seq[i+1] == "g":
                positions.append(i)
    
    if plant:
        #now need to loop through every three nucleotides looking for CpHpG and CpHpH
        for i, nuc in enumerate(seq[:-2]):
            if nuc == "C" or nuc == "c":
                if seq[i+1] != "G" or seq[i+1] != "g": #this takes into account both cases
                    positions.append(i)
    
    return list(set(positions))

def MakeConvertedBaits(base, seq, indexes, n=100):
    SeqGroup = []
    headerGroup = []
    #the powerset can be potentially too large, therefore, I need to subset it down.
    if 2**len(indexes) > n:
        #generate random numbers and only make those iterations, up to n numbers
        #print(f"Making up to {n} iterations")
        np.random.seed(15)
        chosen = np.random.choice(range(2**len(indexes)), size=n,replace=False)
    #iterate over the powerset of the indexes that need to be changed, effectively making all possibilities
        count = 0

        for i in range(0,len(indexes)+1):
            iteration = 0
            for combo in itertools.combinations(indexes, i):
                count += 1
                if count in chosen:
                    iteration += 1
                    splitSeq = [char for char in seq] #make a split version of the nucleotides to replace each index
                    toReplace = combo
                    for index in toReplace:
                        
                        if splitSeq[index] == "c":
                            splitSeq[index] = 't'
                        elif splitSeq[index] == "C":
                            splitSeq[index] == "T"
                    SeqGroup.append("".join(splitSeq))
                    headerGroup.append(base + "-Round{}-Iteration{}-{}-methylations".format(i,iteration,len(combo)))
                else:
                    pass
            
    else:
        for i in range(0,len(indexes)+1):
            iteration = 0
            for combo in itertools.combinations(indexes, i):
                
                iteration += 1
                splitSeq = [char for char in seq] #make a split version of the nucleotides to replace each index
                toReplace = combo
                for index in toReplace:
                    
                    if splitSeq[index] == "c":
                        splitSeq[index] = 't'
                    elif splitSeq[index] == "C":
                        splitSeq[index] == "T"
                SeqGroup.append("".join(splitSeq))
                headerGroup.append(base + "-Round{}-Iteration{}-{}-methylations".format(i,iteration,len(combo)))

    return headerGroup, SeqGroup

def MakeAllPossibleConversionSchemes(baitHeaders, baitSequences, iterations, plant=False ):
    '''Makes all iterations of the baits based on potential methylation patterns
    
    args: 
        baitHeaders : a list of bait headers
        baitSequences : a list of bait sequences
        plant : a bool that determines whether or not the additional methylation schemes CpHpG and CpHpH ; where H = A or T or C  
    
    returns:
        a new list of sequences that include all of the potential methylation schemes and their corresponding headers
    '''
    convertedHeaders = []
    convertedSeqs = []
    
    for header, seq in zip(baitHeaders, baitSequences):
        #produce the complement sequence as we need to do both strands
        comp = ReverseComplementSequence(seq)
        
        if plant:
            forwardIndexes = FindConversionIndexes(seq, plant=True)
            reverseIndexes = FindConversionIndexes(comp, plant=True)
        else:
            forwardIndexes = FindConversionIndexes(seq)
            reverseIndexes = FindConversionIndexes(comp)

        convertForHead, convertForSeqs = MakeConvertedBaits(header, seq, forwardIndexes, iterations)
        convertRevHead, convertRevSeqs = MakeConvertedBaits(header + "-TRA", comp, reverseIndexes, iterations)
        
        convertedHeaders.append(convertForHead)
        convertedHeaders.append(convertRevHead)
                
        convertedSeqs.append(convertForSeqs)
        convertedSeqs.append(convertRevSeqs)
        
        

    return convertedHeaders, convertedSeqs



def CalcHammingDistances(seq1, seq2):
    if len(seq1) != len(seq2):
        raise ValueError("The lengths of the sequences must be the same")
    hamming_dist = hamming(Sequence(seq1), Sequence(seq2))
    if isnan(hamming_dist):
        return 0.0
    else:
        return hamming_dist

def CreateDistanceMatrix(convertedHeaders, convertedSequences):
    matrix = DistanceMatrix.from_iterable(convertedSequences,metric=CalcHammingDistances,keys=convertedHeaders)

    return matrix

def kMedoids(D, k, tmax=100):
	'''Copied this from Matt Johnson who copied code from: https://github.com/letiantian/kmedoids'''
	# determine dimensions of distance matrix D
	m, n = D.shape

	# randomly initialize an array of k medoid indices
	M = np.sort(np.random.choice(n, k))

	# create a copy of the array of medoid indices
	Mnew = np.copy(M)

	# initialize a dictionary to represent clusters
	C = {}
	for t in range(tmax):
		# determine clusters, i. e. arrays of data indices
		J = np.argmin(D[:,M], axis=1)
		for kappa in range(k):
			C[kappa] = np.where(J==kappa)[0]
		# update cluster medoids
		for kappa in range(k):
			J = np.mean(D[np.ix_(C[kappa],C[kappa])],axis=1)
			j = np.argmin(J)
			Mnew[kappa] = C[kappa][j]
		np.sort(Mnew)
		# check for convergence
		if np.array_equal(M, Mnew):
			break
		M = np.copy(Mnew)
	else:
		# final update of cluster memberships
		J = np.argmin(D[:,M], axis=1)
		for kappa in range(k):
			C[kappa] = np.where(J==kappa)[0]

	# return results
	return M, C

def WriteFileForTMCalc(sequences):
    '''Takes in a list of lists of converted sequences and writes every cross iteration to a file for TM calculation'''
    if os.path.exists(os.path.join(os.getcwd(),"temp.txt")):
        fname = "temp-2676236.txt"
    else:
        fname = "temp.txt"
    with open(fname, 'w') as f:
        for seqgroup in sequences:
            if len(seqgroup) < 4:
                pass
            else:
                for i, seq in enumerate(seqgroup):
                    for k,s in enumerate(seqgroup):
                        print(seq)
                        f.write(seq + '\t' + s + '\n')
    return fname

def CallTMCalc(fname, threads):
    os.system(f"java -jar /tools/javatools/GetTmFromSeqPairsMultiThreaded.jar {fname} 50 0.0164 {threads} > TMcalc.txt")

def valuesToDistance(values):
    lines = []
    currentSeq = values[0].split("\t")[0]
    headers = [currentSeq]
    currentLine = ""
    first = True
    for line in values:
        if line.split('\t')[0] != currentSeq:
            lines.append(currentLine + '\n')
            currentSeq = line.split('\t')[0]
            headers.append(line.split('\t')[0])
            currentLine = "{:.3f}".format(float(line.split('\t')[-1].strip()))
        else:
            if first:
                currentLine += "{:.3f}".format(float(line.split('\t')[-1].strip()))
                first = False
            else:
                currentLine += '\t' + "{:.3f}".format(float(line.split('\t')[-1].strip()))
    lines.append(currentLine + '\n')
    return lines, headers

def WriteMatrixToFile(lines, headers, fname):
    with open(fname, 'w') as f:
        f.write("\t".join(headers) + '\n')
        for i in lines:
            f.write(i)

def ReadTMToMatrix(fname, convertedSeqs, convertedHeaders):
    with open(fname, 'r') as f:
        lines = f.readlines()
        counter = 0
        for group, head in zip(convertedSeqs, convertedHeaders):
            if len(group) >= 4:
                base = head[0] + '.matrix'
                toTake = len(group) * len(group)
                values = lines[counter:counter + toTake]
                counter += toTake
                toPrint, headers = valuesToDistance(values)
                print(len(headers), len(toPrint))
                WriteMatrixToFile(toPrint, headers, base)
            else:
                counter += len(group) * len(group)
            break

def ReadInTMDistanceMatrix(fname):
    data = np.genfromtxt(fname, dtype=float, delimiter="\t", skip_header=1)
    d = pairwise_distances(data, metric="precomputed") 
    f = open(fname, 'r')
    lines = f.readlines()
    seqs = [item for item in lines[0].split('\t')]
    return seqs, d


def Main(args):
    headers, seqs = GatherInput(args.infile)
    if args.plant:
        convertedHeaders, convertedSeqs = MakeAllPossibleConversionSchemes(headers, seqs, args.iterations, plant=True)
    else:
        convertedHeaders, convertedSeqs = MakeAllPossibleConversionSchemes(headers, seqs, args.iterations)
    
    count = 0
    for heads, s in zip(convertedHeaders, convertedSeqs):
        if len(heads) == 1 and len(s) == 1:
            pass
        else:
            count += 1
        #    matrix = DistanceMatrix.from_iterable(s, metric=CalcHammingDistances, keys=heads)
    
    #for now need to write a file to calculate all of the TMs for the possibilities. This file will likely be huge
    if args.TM:
        fileName = WriteFileForTMCalc(convertedSeqs)
        CallTMCalc(fileName, args.threads)
        ReadTMToMatrix("TMcalc.txt", convertedSeqs, convertedHeaders)

        for f in os.listdir(os.getcwd()):
            if f.split('.')[-1] == "matrix":
                seqs, matrix = ReadInTMDistanceMatrix(f)
                for i in range(2,7):
                    try:
                        medoids, clusters = kMedoids(matrix, i)
                        print(medoids, clusters)
                        for med in medoids:
                            print(seqs[med])
                            print(matrix[med])
                    except ValueError:
                        pass

    else:
        count = 0
        for heads, s in zip(convertedHeaders, convertedSeqs):
            if len(heads) == 1 and len(s) == 1:
                pass
        else:
            print("got to here")
            count += 1
            matrix = DistanceMatrix.from_iterable(s, metric=CalcHammingDistances, keys=heads)
            for i in range(2,7):
                try:
                    medoids, clusters = kMedoids(matrix, i)
                    print(medoids, clusters)
                except ValueError:
                    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Makes baits for methyl cap from an existing bait set produced with BaitDesigner. It will produce all possible combinations of baits and then attempt to reduce the number via a k-medoids clustering approach")

    parser.add_argument("infile", help="A bait file produced with BaitDesigner")
    parser.add_argument("--plant", action="store_true", help="If this is for a plant genome, the additional methylation schemes (apart from CpG) CpHpG and CpHpH will be considered")
    parser.add_argument("--threads", nargs="?", const="4", help=" Default value: 4 . How many threads you would like to run for TM calculation")
    parser.add_argument("--iterations", nargs="?", const=100,type=int, help="How many methyl schemes to predict. This is greatly constrained by amount of RAM available on the machine. 100 is a good a place to start, 5000 is likely too large. The larger this number is the better chances your medoids are actual medoids though. Its a delicate balance.")
    parser.add_argument("--TM", action="store_true", help="If you would like the disimilarity matrix (and thus the clustering) to be based on the TM of possible probe choices then provide this option.")

    args = parser.parse_args()

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)

    Main(args)

