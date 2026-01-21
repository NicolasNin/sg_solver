from board_detection.cnns import *
from board_detection.graph_clustering import *
from board_detection.reference_data import *
import numpy as np
import joblib
import time
# current models
rf_model_compare_patch = joblib.load('/home/nicolas/code/star_genius_solver/data/models/random_forest_model.pkl')
color_model =  joblib.load('/home/nicolas/code/star_genius_solver/data/models/color_classifier_knn2.pkl')

state_dict_reverse = torch.load('/home/nicolas/code/star_genius_solver/data/emptyAttention_onbig_reverse_best.pth', weights_only=True)
modelCnnEmpty = EmptyNetAttention(in_channels=3).cuda()
modelCnnEmpty.load_state_dict(state_dict_reverse)

#state_dict_reverse = torch.load('/home/nicolas/code/star_genius_solver/data/emptyAttention3.pth', weights_only=True)
#modelCnnEmpty = EmptyNetAttention(in_channels=3,N=2,embedding_dim=128).cuda()
#modelCnnEmpty.load_state_dict(state_dict_reverse)



state_dict_reverse = torch.load('/home/nicolas/code/star_genius_solver/data/multi_onbig_reverse_best.pth', weights_only=True)
modelMulti = EmptyNetMultiTask(in_channels=3, attention_temp=0.5,N=2,embedding_dim=128)
modelMulti.load_state_dict(state_dict_reverse)

def classify_patches_empty(patches_oriented):
    #input is BGR
    t=time.time()
    patches_lab = np.array([cv2.cvtColor(p,cv2.COLOR_BGR2LAB) for p in patches_oriented])
    #print(patches_lab.shape)
    #x = torch.from_numpy(patches_lab).permute(2, 0, 1).float() / 255.0
    x = torch.from_numpy(patches_lab).permute(0,3, 1, 2).float() / 255.0
    x=x.cuda()
    #print(x.shape)
    modelCnnEmpty.eval()
    with torch.no_grad():
        logits = modelCnnEmpty(x)
        probs = torch.sigmoid(logits)
    #print([f"{float(x[0]):.3}" for x in probs])
    print(f"classification of empty took {time.time()-t}")
    return [x>0.5 for x in probs],probs.cpu().numpy().squeeze()

def classify_patches_multi_from_oriented(patches_oriented):
    #BGR patches input
    modelMulti.eval()
    t=time.time()
    patches_oriented_lab =[]
    for i,p_oriented in enumerate(patches_oriented):
        p_lab  =cv2.cvtColor(p_oriented,cv2.COLOR_BGR2LAB) 
        patches_oriented_lab.append(p_lab)
    x = torch.from_numpy(np.array(patches_oriented_lab)).permute(0,3, 1, 2).float() / 255.0
    logits = modelMulti(x)
    probs = {}
    for ta in logits:
        probs[ta] = torch.sigmoid(logits[ta]).cpu().detach().numpy()[:,0]
    return probs


def classify_patches_multi(patches):
    #BGR patches input
    modelMulti.eval()
    t=time.time()
    patches_oriented_lab =[]
    for i,p in enumerate(patches):
        orientation = TRIANGLES_ORIENTATION[i+1]
        p_oriented = p if orientation == "up" else  cv2.rotate(p, cv2.ROTATE_180)
        p_lab  =cv2.cvtColor(p,cv2.COLOR_BGR2LAB) 
        patches_oriented_lab.append(p_lab)
    x = torch.from_numpy(np.array(patches_oriented_lab)).permute(0,3, 1, 2).float() / 255.0
    logits = modelMulti(x)
    probs = {}
    for ta in logits:
        probs[ta] = torch.sigmoid(logits[ta]).cpu().detach().numpy()[:,0]
    return probs


def compute_all_compare(color_data):
    feats = []
    for i in range(1,49):
        for j in range(i+1,49):
            cd1 = color_data[i]
            cd2 = color_data[j]
            feat = np.array(distance_color_data(cd1,cd2))
            feats.append(feat)
    feats = np.array(feats)
    feats.shape
    compare_vals = rf_model_compare_patch.predict_proba(feats)[:,1]
    mat_up = np.triu(np.ones(48*48).reshape((48,48)),k=1).astype(bool)
    mat = np.zeros((48,48))
    mat[mat_up]=compare_vals
    mat = mat+mat.T
    np.fill_diagonal(mat, 1)
    return mat

def softmax(logits, axis=-1):
    """
    Numerically stable softmax along given axis.
    logits: array of shape (..., K)
    """
    x = logits - np.max(logits, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def unary_from_two_binary(p_empty, p_white, eps=1e-6):
    """
    p_empty: (N,)  ~ P(E | x)
    p_white: (N,)  ~ P(W | x)

    Returns P_unary: (N, 3) with columns [P(E), P(W), P(C)].
    """
    pE = np.clip(p_empty, 0.0, 1.0)
    pW = np.clip(p_white, 0.0, 1.0)
    pC = 1.0 - pE - pW

    # Clip negatives if the two nets are inconsistent (sum > 1)
    pC = np.clip(pC, 0.0, 1.0)

    P = np.stack([pE, pW, pC], axis=-1)
    S = np.sum(P, axis=-1, keepdims=True)
    S = np.clip(S, eps, None)
    P /= S
    return P

def refine_with_similarity(P_unary,
                           S,
                           lam=1.0,
                           n_iters=10,
                           eps=1e-6,
                           normalize_S=True):
    """
    Refine per-patch label probabilities using similarity-based smoothing.

    We look for a fixed point Q such that, for each patch i and label y:
        Q_i(y) ∝ exp( log P_unary(i,y) + lam * sum_j S[i,j] * Q_j(y) )

    Shapes
    ------
    P_unary: (N, 3) array, rows sum to 1. (from build_unary_potential)
    S      : (N, N) similarity matrix. Larger S[i,j] => stronger influence.

    Parameters
    ----------
    lam         : float, weight of similarity term (λ in our discussion).
    n_iters     : number of fixed-point iterations.
    normalize_S : if True, row-normalize S so each row sums to 1.

    Returns
    -------
    Q      : (N, 3) refined probabilities.
    y_hat  : (N,) hard labels in {0,1,2} for {E,W,C}.
    """

    N, K = P_unary.shape
    assert K == 3, "Expecting 3 labels: E, W, C"

    # Optionally row-normalize S so each node's neighbor influence sums to 1.
    if normalize_S:
        row_sums = np.sum(S, axis=1, keepdims=True)
        # Avoid division by zero for isolated patches
        row_sums = np.clip(row_sums, eps, None)
        S_norm = S / row_sums
    else:
        S_norm = S

    # Initialize beliefs with unary probabilities
    Q = P_unary.copy()

    # Precompute log P_unary; this term stays fixed over iterations.
    log_P_unary = np.log(np.clip(P_unary, eps, 1.0))

    for t in range(n_iters):
        # NeighborSupport(i,y) = sum_j S_norm[i,j] * Q_j(y)
        # Using matrix multiply: (N,N) @ (N,3) -> (N,3)
        neighbor_support = S_norm @ Q  # shape (N, 3)

        # NewScore(i,y) = log P_unary(i,y) + lam * NeighborSupport(i,y)
        new_scores = log_P_unary + lam * neighbor_support

        # Update Q with softmax so rows remain probabilities
        Q = softmax(new_scores, axis=1)

        # (Optional) You can check convergence here if you want:
        # max_change = np.max(np.abs(Q - Q_prev))
        # and break early if max_change < some_threshold.

    # Hard labels: argmax over {E, W, C}
    y_hat = np.argmax(Q, axis=1)

    return Q, y_hat


def compute_normalized_lab(color_data,white_ids,L_ref=90.0,a_ref=0.0,b_ref=0.0):
    #this is for the color model input
    #white_ids start at 0 and color_data doest not have normalized L
    L_meds = np.array([color_data[idx]["L_med"]/255*100 for idx in color_data]) #
    A_meds = np.array([color_data[idx]["A_med"] for idx in color_data])
    B_meds = np.array([color_data[idx]["B_med"] for idx in color_data])
    if len(white_ids)==0:
        print(f"Warning no white found to normalize colors")
        dL=0
        da=0
        db=0
    else:
        white_L = np.mean(L_meds[white_ids])
        white_A= np.mean(A_meds[white_ids])
        white_B = np.mean(B_meds[white_ids])
        dL = L_ref - white_L
        da = a_ref - white_A
        db = b_ref - white_B
    return np.array([L_meds+dL,A_meds+da,B_meds+db]).T

def rescale_pw(pW_raw):
    #pW_raw = probs_dict["white"]    
    # with "bad" white classifier      
    w_max = max(0.7,np.percentile(pW_raw, 99))
    pW = np.clip(pW_raw / max(w_max, 1e-6), 0, 1)
    return pW

def compute_empty_white_colored(pE,pW,S):
    #pE = probs_empty         

    #P_unary = build_unary_potential( p_empty=pE,p_white=pW, alpha_empty=1.0, alpha_white=1.0,)
    P_unary = unary_from_two_binary(pE, pW, eps=1e-6)
    Q, y_hat = refine_with_similarity( P_unary=P_unary, S=S, lam=1.0, n_iters=10,)
    return Q,y_hat