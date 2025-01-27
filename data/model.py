import torch.nn as nn
import torch
from torch.nn import functional as F
import math
from transformers import TimesformerForVideoClassification

class TransformerForSingleGlossPrediction(nn.Module):
    def __init__(self):
        super(TransformerForSingleGlossPrediction, self).__init__()

        self.transformer = nn.Transformer(
            d_model=128,
            nhead=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=64,
            dropout=0.1,
        )

 
        self.positional_encoding = nn.Embedding(15, 128)

        self.output_layer = nn.Linear(1920 , 256)

    def forward(self, x):


        seq_len_x = x.size(0)
        x_pos = self.positional_encoding(torch.arange(0, seq_len_x, device=x.device).unsqueeze(1).expand(-1, x.size(1)).long())

        x_input = x + x_pos

        transformer_output = self.transformer(x_input, x_input)

        batch_size,num_frames,features = transformer_output.size()# [2, 10, 1280]
        pooled_output = transformer_output.reshape(batch_size,num_frames*features)
        # print("after reshape in the transformer: ", pooled_output.shape)

        output = self.output_layer(pooled_output)

        return output

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        

        self.model = nn.Sequential(
            nn.Conv2d(3,32,3),
            nn.Conv2d(32,32,3),
            nn.Conv2d(32,64,3),  
            nn.MaxPool2d(5),
            nn.Conv2d(64,128,5),
            nn.MaxPool2d(5),  
            nn.Conv2d(128,128,3),
            nn.MaxPool2d(3),  
  
       
        )
        self.dropout = nn.Dropout(0.1)
        self.transformer = TransformerForSingleGlossPrediction()
        self.fc1 = nn.Linear(2176 ,64)
        self.fc4 = nn.Linear(64,2)

    def forward(self,x):
        batch_size, num_frames, channels, height, width = x.size()
        x = x.reshape(batch_size*num_frames,channels,height,width)
        extracted_feature = self.model(x)
        # print("x shape after model: ", x.shape)
        x = extracted_feature.reshape(batch_size,-1)
        x = x.reshape(batch_size,num_frames,-1)
        # print("x shape after reshape: ", x.shape)
        transformer_output = self.transformer(x)
        extracted_feature = extracted_feature.reshape(batch_size,-1)
        x = torch.cat((transformer_output, extracted_feature), dim=1)
      
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return F.softmax(self.fc4(x))