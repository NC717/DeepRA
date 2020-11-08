# DeepRA - Predicting Joint Damage from Radiographs Using CNN with Attention
Joint damage in Rheumatoid Arthritis (RA) is assessed by manually inspecting and grading radiographs of hands and feet. This is a tedious task which requires trained experts
whose subjective assessment leads to low inter-rater agreement. An algorithm which can learn the hidden representations from the radiographs and predict the joint level damage
in hands and feet can help optimize this process which will eventually aid the doctors in better patient care and research. In this paper, we propose a two staged approach which
amalgamates object detection and convolutions with attention which can efficiently and accurately predict the overall and joint level erosion and narrowing from patients radiographs.
This approach has been evaluated on hand and feet radiographs of patients suffering with RA and has achieved a weighted root mean squared error (RMSE) of 0.442 and 0.49 in predicting joint level erosion and narrowing Sharp/van der Heijde (SvH) scores and a weighted absolute error of
0.542 while predicting the overall damage in hand and feet radiographs for patients. The proposed approach was developed during the RA2 Dream Challenge hosted by dream challenges
1 and secured 4th and 8th position in predicting overall and joint level erosion and narrowing SvH scores from radiographs respectively.

# Sample output from the Object Detection Model for Hand and Feet
![alt text](https://github.com/NC717/DeepRA/blob/main/images/Final_foot_annotation.JPG?raw=true "Fingers detected in feet radiographs")
![alt text](https://github.com/NC717/DeepRA/blob/main/images/Final_hand_annotation.JPG?raw=true "Fingers/Wrist detected in hand radiographs")

# Attention Map visuals
- Foot Erosion Attention Maps
![alt text](https://github.com/NC717/DeepRA/blob/main/images/Foot_erosion_attention_maps.JPG?raw=true "Attention weights while predicting erosion in feet fingers")
- Wrist Erosion Attention Maps
![alt text](https://github.com/NC717/DeepRA/blob/main/images/Wrist_erosion_attention.JPG?raw=true "AAttention weights while predicting erosion in wrist")
