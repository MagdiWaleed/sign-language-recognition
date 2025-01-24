import cv2
import numpy as np 
import matplotlib.pyplot as plt
import torch.nn.functional as F # type: ignore
from multiprocessing import Process, Manager
import torch


#72
# np.linspace

class GlossesRecognition():
    def __init__(self,model,num_frames:int,dictanory:dict,device: str = 'cpu',maximum_glosses_length: int =3,length_of_frames_predictions= 72):
        
        # groups
        self.totalframes = []
        self.previous_glosses_array = []
        self.displayed_gloss_text = ""
        self.total_glosses_counter = 0 #to tell us how many glossess we must try to predict

        self.finished = 0
        
        #reading variables
        self.reading_counter = 0

        #showing variables 
        self.showing_counter  = 0
        self.maximum_glosses_length = maximum_glosses_length
        
        #gloss predictions
        self.model = model
        self.num_frames = num_frames
        self.dictanory = dictanory
        self.device = device
        self.length_of_frames_predictions = length_of_frames_predictions
        self.current_glosses_counter = 0 #to tell us we predict the i-th gloss (note this variable and total_glosses_counter)  

        self.indcies_range = np.linspace(0,self.length_of_frames_predictions-1,self.num_frames,dtype=np.int32)
    def start_recognition(self):
        with Manager() as manager:

            self.totalframes = manager.list(self.totalframes) 
            # self.previous_glosses_array = []
            # self.displayed_gloss_text = ""
            self.total_glosses_counter = manager.Value('i',self.total_glosses_counter) #to tell us how many glossess we must try to predict
            
            #reading variables
            self.reading_counter = manager.Value('i',self.reading_counter)

            #showing variables 
            self.showing_counter  = manager.Value('i',self.showing_counter)  #to tell us we predict the i-th gloss (note this variable and total_glosses_counter)  
            # self.maximum_glosses_length = maximum_glosses_length
            
            #gloss predictions
            # self.model = model
            # self.num_frames = num_frames
            # self.dictanory = dictanory
            # self.device = device
            # self.length_of_frames_predictions = length_of_frames_predictions
            self.current_glosses_counter = manager.Value('i',self.current_glosses_counter)
            self.finished = manager.Value('i',self.finished)
            reading = Process(target=self.readFrames,)
            showing = Process(target=self.showFrames,)
            predict_glosses = Process(target=self.getGlossPredictions,)


            reading.start()
            showing.start()
            predict_glosses.start()

            reading.join()
            showing.join()
            predict_glosses.join()

    def readFrames(self): 
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("somthing went wrong")
            exit()
 
        while self.reading_counter.value<1000:
            
            ret, frame = cap.read()
            if not ret:
                print("somthing went wrong while reading frames")
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            self.totalframes.append(frame)
            self.reading_counter.value += 1
            if self.reading_counter.value %self.length_of_frames_predictions == 0:
                self.total_glosses_counter.value+=1
            if self.finished.value == 1:
                break

    def showFrames(self):
        while True: 

            if self.reading_counter.value ==0 and self.showing_counter.value ==0:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
            elif self.showing_counter.value < self.reading_counter.value:
                frame = self.totalframes[self.showing_counter.value-1]
            elif self.reading_counter.value == 0:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                frame = self.totalframes[self.reading_counter.value-1]

            cv2.putText(
                    frame,"sh-"+str(self.current_glosses_counter.value+1 )+", rd-"+str(self.total_glosses_counter.value+1),
                    (10, 50) ,cv2.FONT_HERSHEY_SIMPLEX,0.8, 
                    (0, 255, 0), 1,lineType=cv2.LINE_AA)# position  #font_scale #thickness


            cv2.imshow("Video", frame)
            key = cv2.waitKey(1)  # Wait ~30ms per frame (adjust for your FPS)

            self.showing_counter.value += 1
           
            if key == 27:  # ESC key
                print("Exiting video playback.")
                self.finished.value = 1
                break

        cv2.destroyAllWindows()
    def getGlossPredictions(self):
        while True:
            if self.finished.value == 1:
                break
            if self.current_glosses_counter.value < self.total_glosses_counter.value:
                predict_this= np.array(self.totalframes[-self.length_of_frames_predictions:])
                current_gloss_video = predict_this[self.indcies_range]
                # print(current_gloss_video.shape)

                gloss_probabilities = self.predict(self.model,[current_gloss_video],self.device)
                gloss_probabilities = F.softmax(gloss_probabilities, dim=1).cpu().detach().numpy()
                gloss_string, displayed_gloss = self.get_gloss_string(gloss_probabilities,self.dictanory,self.previous_glosses_array)
                self.previous_glosses_array.append(gloss_string)
                self.displayed_gloss_text = displayed_gloss
                self.current_glosses_counter.value += 1
           

                print(f"Predicted gloss: {gloss_string}, showing_c: {self.showing_counter.value}, reading_c: {self.reading_counter.value}")
    def predict(self,model,frames,device):
        frames = torch.tensor(frames).clone().detach()
        frames = frames.permute(0,1, 4, 2, 3).float().to(device) # batch_size, num_frames, channels, height, width
        frames = frames.float()
        outputs = model(frames)
        return outputs
        
        
    def get_gloss_string(self,gloss_probabilities,dictanory,previous_glosses_array:list):
        gloss_index = np.argmax(gloss_probabilities)
        gloss_string = dictanory[str(gloss_index)]
        displayed_gloss= ''

        if len(previous_glosses_array) < 3:
            for gloss in previous_glosses_array:
                displayed_gloss += gloss + ', '
        else:
            displayed_gloss =previous_glosses_array[-3] +', '+ previous_glosses_array[-2] +', '+ previous_glosses_array[-1] +', '
            displayed_gloss+=gloss_string    

        return gloss_string, displayed_gloss


            #     cv2.imshow('webcame video',self.totalframes[self.showing_counter])

              

# def get_gloss_string(gloss_probabilities,dictanory,previous_glosses_array:list):
#     gloss_index = np.argmax(gloss_probabilities)
#     gloss_string = dictanory[str(gloss_index)]
#     displayed_gloss= ''

#     if len(previous_glosses_array) < 3:
#         for gloss in previous_glosses_array:
#             displayed_gloss += gloss + ', '
#     else:
#         displayed_gloss =previous_glosses_array[-3] +', '+ previous_glosses_array[-2] +', '+ previous_glosses_array[-1] +', '
#         displayed_gloss+=gloss_string    

#     return gloss_string, displayed_gloss
    




            # height, width, _ = self.totalframes[self.showing_counter].shape  # Get frame dimensions
            
            # if counter % 5 == 0 and counter <= num_frames+1:
                
            #     if len(main_text) !=3 :
            #         main_text += "."
            #     else:
            #         main_text = ''
            # else:
            #     main_text = displayed_glosses_text
            
            # main_text_font = cv2.FONT_HERSHEY_SIMPLEX
            # main_text_font_scale = 1
            # main_text_color = (255, 0, 0)  # Blue
            # main_text_thickness = 3
            
            # text_size = cv2.getTextSize(main_text, main_text_font, main_text_font_scale, main_text_thickness)[0]
            # text_width = text_size[0]

            # main_text_position = ((width - text_width) // 2, height - 30) 


            # cv2.putText(frame, main_text, main_text_position, main_text_font, main_text_font_scale, main_text_color, main_text_thickness, lineType=cv2.LINE_AA)


            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     print("Quitting the video stream")
            #     break
    
    # print("dictanory",dictanory)
    # print(gloss_index)