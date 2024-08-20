from trigger import category_freq, load_trigger_images
from embedding import get_emb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader, TensorDataset
from options import get_parser_args
from sklearn.decomposition import PCA


def cal_sim(vector_0, vector_1):
    
    '''
    Calculate the cos sim and pairwise distance
    :param vector_0:
    :param vector_1:
    :return: cos_sim, pair_dis
    '''
    cos_sim_f = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    pair_dis_f = torch.nn.PairwiseDistance(p=2)
    cos_sim = cos_sim_f(vector_0, vector_1)
    pair_dis = pair_dis_f(vector_0, vector_1)
    return cos_sim, pair_dis
if __name__== "__main__" :
    
    
    parser = get_parser_args()
    file_path = '/root/trigger'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Define the dimension of the embeddings
    d = 768  # as specified in the request
    client_num = 5
    W = []
    num = parser['trigger_num']
    # Define the hyperparameter beta
    beta = 0.05

    # Assuming Y and T are your image and text embedding sets respectively
    # Y and T should be tensors of shape [batch_size, d, m]
    Y, T =  get_emb(client_num,f'/root/autodl-tmp/trigger-images-wo-noise-{num}')
    method = parser['method']

    for i in range(client_num):
        print(f'client {i}')
        Trigger_mat_pth = os.path.join(file_path, f"/root/trigger/results/w_matrix/noisy_w/trigger_mat_c{i}_{num}_{method}.pth" )
        if os.path.exists(Trigger_mat_pth):
            W_align = torch.load(Trigger_mat_pth)
            W_align = W_align.to(device)
            print("Loaded Trigger matrix from", Trigger_mat_pth)
        else:
            
            dataset = TensorDataset(Y[i], T[i])
            if num > 64:
                batch_size =64
            else:
                batch_size = num
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            # Randomly initialize the linear transformation matrix W_align
            W_align = torch.randn(d, d, requires_grad=True, device=device)
            torch.nn.init.orthogonal_(W_align)
        
            # Define the optimizer
            optimizer = optim.Adam([W_align], lr=1e-3, eps=1e-5)

            # Training loop
            num_epochs = 1000 # Set number of epochs as needed
            for epoch in range(num_epochs):
                for batch_image_embeddings, batch_text_embeddings in data_loader:
                    optimizer.zero_grad()
    
                    # Calculate the loss function as ||W_align * Y - W_align.T * T||_F
                    loss = torch.norm(W_align @ batch_image_embeddings.T - W_align @ batch_text_embeddings.T, p='fro')
    
                    # Backpropagate the loss
                    loss.backward()
    
                   
                    # torch.nn.utils.clip_grad_norm_(W_align, max_norm=0.5)
                    torch.nn.utils.clip_grad_value_(W_align, clip_value=0.5)
                    optimizer.step()
                     
                    if parser['method'] != 'random':
                    # Manually update W_align with the orthogonal constraint
                        with torch.no_grad():
                            W_align -= beta * (W_align @ W_align.T @ W_align - W_align)

                if epoch % 10 == 0:  # Print loss every 10 epochs
                    print(f'Epoch {epoch}, Loss: {loss.item()}')
                
            torch.save(W_align, f'/root/trigger/results/w_matrix/noisy_w/trigger_mat_c{i}_{num}_{method}.pth')
            W.append(W_align)
            print("Training complete.")
        
        # watermark verification
        origin_image_features = Y[i]
        origin_text_features = T[i]
        
        # watermark
        image_features = origin_image_features @ W_align.T
        text_features = origin_text_features @ W_align.T

        images_emb = image_features / image_features.norm(dim=-1, keepdim = True)
        texts_emb = text_features / text_features.norm(dim=-1, keepdim = True) 
  
        if origin_image_features.shape[0] == origin_text_features.shape[0]:
            
            origin_cos_sim, origin_pair_dis = cal_sim(origin_image_features, origin_text_features)
            
            Trigger_cos_sim, Trigger_pair_dis = cal_sim(images_emb, texts_emb)
            print(origin_pair_dis)
            
            
            
            print("Origin: cos similarity: %lf, pair distance: %lf" % (float(origin_cos_sim.mean()), float(origin_pair_dis.mean())))
            print("Trigger_mat: cos similarity: %lf, pair distance: %lf" % (float(Trigger_cos_sim.mean()), float(Trigger_pair_dis.mean())))
            
            
            
            print('delta cos similarity: %lf' % ((float(Trigger_cos_sim.mean())-float(origin_cos_sim.mean()))))
            print('delta pair distance: %lf' % (-float(Trigger_pair_dis.mean())+float(origin_pair_dis.mean())))
            
            m_cos = (((float(Trigger_cos_sim.mean())-float(origin_cos_sim.mean())))+2)/4
            m_l2 = ((-float(Trigger_pair_dis.mean())+float(origin_pair_dis.mean()))+2)/4
            
            print('delta total:', 0.5*m_cos+0.5*m_l2)
            
           
            


