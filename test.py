from multiprocessing import Process , Manager
import numpy as np
class hi:
    def __init__(self):
        self.frames =[np.random.normal(1,1,size=(224, 224, 3)) ]
    def print_frame(self):
        while True:
            print(np.array(self.frames).shape)
    def add_frame(self):
        while True:
            self.frames.append(np.random.normal(1,1,size=(224, 224, 3)))

    def start(self):
        with Manager() as manager:
            self.frames = manager.list(self.frames)  # use Manager's list to share the list between processes
            process1 = Process(target=self.add_frame, )
            process2 = Process(target=self.print_frame,)
            process1.start()
            process2.start()
            process1.join()
            process2.join()
            

if __name__ == "__main__":
    hi().start()
    