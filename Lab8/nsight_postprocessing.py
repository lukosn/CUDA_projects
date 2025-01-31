import csv
import argparse


def main(filename : str) -> None:
    globalQueueData = dict()
    blockQueueData = dict()
    with open(filename) as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)
        statKeys = header[7:11]
        for k in statKeys:
            globalQueueData[k] = list()
            blockQueueData[k] = list()
        for row in csvreader:
            if row[1] == 'BFSKernelGlobalQueue':
                for k in statKeys:
                    globalQueueData[k].append(float(row[header.index(k)].replace(",","")))
            elif row[1] == 'BFSKernelBlockQueue':
                for k in statKeys:
                    blockQueueData[k].append(float(row[header.index(k)].replace(",","")))
    maxGlobal = dict()
    maxBlock = dict()
    for k in statKeys:
        maxGlobal[k] = max(globalQueueData[k])
        maxBlock[k] = max(blockQueueData[k])
    totalTimeGlobal = sum(globalQueueData[statKeys[1]])
    totalTimeBlock = sum(blockQueueData[statKeys[1]])
    print("--- Average values ---")
    print(maxGlobal)
    print(maxBlock)
    print("--- Total execution time ---\nGlobal: {globalTime:.2f}\nBlock: {blockTime:.2f}".format(globalTime=totalTimeGlobal, blockTime=totalTimeBlock))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = "Nsight Postprocessing")
    parser.add_argument("filename")
    
    args = parser.parse_args()
    
    main(args.filename)