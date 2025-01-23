import cv2
import numpy as np 
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F # type: ignore
def readFrames(model,num_frames:int,dictanory:dict,device: str = 'cpu',maximum_glosses_length: int =3):
    
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("somthing went wrong")
        exit()
    counter = 1
    total_frames = []
    gloses_array = []
    displayed_glosses_text = ""
    main_text = "."
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("somthing went wrong while reading frames")
                break
            
            

            cv2.putText(
                frame,str(counter+1 ),
                (10, 50) ,cv2.FONT_HERSHEY_SIMPLEX,1, 
                (0, 255, 0),1,lineType=cv2.LINE_AA)# position  #font_scale #thickness

            height, width, _ = frame.shape  # Get frame dimensions
            
            if counter % 5 == 0 and counter <= num_frames+1:
                
                if len(main_text) !=3 :
                    main_text += "."
                else:
                    main_text = ''
            else:
                main_text = displayed_glosses_text
            
            main_text_font = cv2.FONT_HERSHEY_SIMPLEX
            main_text_font_scale = 1
            main_text_color = (255, 0, 0)  # Blue
            main_text_thickness = 3
            
            text_size = cv2.getTextSize(main_text, main_text_font, main_text_font_scale, main_text_thickness)[0]
            text_width = text_size[0]

            main_text_position = ((width - text_width) // 2, height - 30) 


            cv2.putText(frame, main_text, main_text_position, main_text_font, main_text_font_scale, main_text_color, main_text_thickness, lineType=cv2.LINE_AA)


            cv2.imshow('webcame video',frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            total_frames.append(frame)
            
            if counter>num_frames:
                # print("we are in ")
                gloss_probabilities = predict(model,[total_frames[-num_frames:]],device)
                gloss_probabilities = F.softmax(gloss_probabilities)
                # print("gloss_probabilities",gloss_probabilities)
                
                # print(f"Gloss proobabilties: {gloss_probabilities}, description: {gloss_probabilities}")
                if len(gloses_array)>maximum_glosses_length:
                    gloses_array = gloses_array[maximum_glosses_length:]

                gloss_string,displayed_glosses= get_gloss_string(gloss_probabilities.cpu().detach().numpy(),dictanory,gloses_array)
                displayed_glosses_text = displayed_glosses
                # print(f"Predicted gloss: {gloss_string}")
                gloses_array.append(gloss_string)
                
                
                total_frames = []
                counter = 0


            counter+=1
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting the video stream")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

def get_gloss_string(gloss_probabilities,dictanory,previous_glosses_array:list):
    gloss_index = np.argmax(gloss_probabilities)
    # print("dictanory",dictanory)
    # print(gloss_index)
    gloss_string = dictanory[str(gloss_index)]
    displayed_gloss= ''

    if len(previous_glosses_array) < 3:
        for gloss in previous_glosses_array:
            displayed_gloss += gloss + ', '
    else:
        displayed_gloss =previous_glosses_array[-3] +', '+ previous_glosses_array[-2] +', '+ previous_glosses_array[-1] +', '
        displayed_gloss+=gloss_string    

    return gloss_string, displayed_gloss
    
def predict(model,frames,device):
    frames = torch.tensor(frames).clone().detach()
    frames = frames.permute(0,1, 4, 2, 3).float().to(device) # batch_size, num_frames, channels, height, width
    frames = frames.float()
    outputs = model(frames)
    return outputs



