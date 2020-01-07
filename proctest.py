import random, time, multiprocessing
 
def func(arg):
    time.sleep( random.uniform(0,2) )
    return '[bla %s]'%arg
 
if __name__ == '__main__':
    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        start = time.time()
        for res in p.imap_unordered(func, range(20), chunksize=2):
            print("(after %3.1fsec)  returnval:%s"%(time.time()-start, res))