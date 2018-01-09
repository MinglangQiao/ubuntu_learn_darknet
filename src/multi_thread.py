import os, time, random
from multiprocessing import Process
from multiprocessing import Pool



def test_1():

    print('Process (%s) start' % os.getpid())
    pid = os.fork()
    # creat two process, both will run the following code
    if pid == 0:
        print('I am child process(%s) and my parent is %s.' % (os.getpid(), os.getppid()))
    else:
        print('I (%s) just creat a child process (%s)' % (os.getpid(), pid))

    print('>>> test process')


def test_2():
    # 子进程要执行的代码
    def run_proc(name):
        print('Run child process %s (%s)...' % (name, os.getpid()))

    if __name__=='__main__':
        print('Parent process %s.' % os.getpid())
        p = Process(target=run_proc, args=('test',))
        print('Child process will start.')
        p.start()
        p.join()
        print('Child process end.')

def test_3():
    # should be a lonly function, not in test_3
    def long_time_task(name):
        print('Run task %s (%s)...' % (name, os.getpid()))
        start = time.time()
        time.sleep(random.random() * 3)
        end = time.time()
        print('Task %s runs %0.2f seconds.' % (name, (end - start)))

    # 'shoud run in main'
    # print('Parent process %s.' % os.getpid())
    # p = Pool(4)
    # for i in range(5):
    #     p.apply_async(long_time_task, args=(i,))
    # print('Waiting for all subprocesses done...')
    # p.close()
    # p.join()
    # print('All subprocesses done.')

def test_4():
    import subprocess

    print('$ nslookup www.python.org')
    r = subprocess.call(['nslookup', 'www.python.org'])
    print('Exit code:', r)

def test_5():
    import subprocess

    print('$ nslookup')
    p = subprocess.Popen(['nslookup'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = p.communicate(b'set q=mx\npython.org\nexit\n')
    print(output.decode('utf-8'))
    print('Exit code:', p.returncode)

def test_6():
    from multiprocessing import Process, Queue
    import os, time, random

    def write(q):
        print('Process to write: %s' % os.getpid())
        for value in ['A', 'B', 'C']:
            print('Put %s to queque...' % value)
            q.put(value)
            time.sleep(random.random())

    def read(q):
        print('Process to read: %s' % os.getpid)
        while True:
            value = q.get(True)
            print('Get %s frome queue.' % value)

    q = Queue()
    pw = Process(target = write, args = (q, ))
    pr = Process(target = read, args = (q, ))

    pw.start()
    pr.start()
    pw.join()
    pr.terminate()


# >>>>>>>>>>>>>>>>>>>> Threading
def test_7():

    import time, threading

    # 新线程执行的代码:
    def loop():
        print('thread %s is running...' % threading.current_thread().name)
        n = 0
        while n < 5:
            n = n + 1
            print('thread %s >>> %s' % (threading.current_thread().name, n))
            time.sleep(1)
        print('thread %s ended.' % threading.current_thread().name)

    print('thread %s is running...' % threading.current_thread().name)
    t = threading.Thread(target=loop, name='LoopThread')
    t.start()
    t.join()
    print('thread %s ended.' % threading.current_thread().name)


def test_7():
    'copy to tempy file to run'
    import time, threading

    # 假定这是你的银行存款:
    balance = 0

    def change_it(n):
        # 先存后取，结果应该为0:
        global balance
        balance = balance + n
        balance = balance - n

    def run_thread(n):
        for i in range(100000):
            change_it(n)

    t1 = threading.Thread(target=run_thread, args=(5,))
    t2 = threading.Thread(target=run_thread, args=(8,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    print(balance)



def run():
    test_7()



if __name__ == "__main__":

    run()
