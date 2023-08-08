import os
from tqdm import tqdm
from multiprocessing import Pool

def run_main(pid):
    os.system(f'python collector.py {pid}')

if __name__ == '__main__':
    num_processes = 32  # 원하는 프로세스 수를 설정하세요
    pool = Pool(num_processes)
    
    results = []
    for i in tqdm(range(16000)):
        results.append(pool.apply_async(run_main, args=(i,)))
    
    pool.close()
    pool.join()
    
    for result in results:
        result.get()
