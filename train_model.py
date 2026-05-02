"""
train_model.py  -  Backpropagation FROM SCRATCH dengan NumPy
Prediksi Risiko Obesitas — Tugas 7 Modul Kecerdasan Buatan
Dataset: obesity.csv (2111 baris, 17 kolom)
"""
import numpy as np
import pandas as pd
import pickle, json, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)
BASE = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(BASE, "obesity.csv"))
print(f"Dataset dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")

cat_cols = ['Gender', 'CALC', 'FAVC', 'SCC', 'SMOKE',
            'family_history_with_overweight', 'CAEC', 'MTRANS']
cat_encoders = {}
for col in cat_cols:
    enc = LabelEncoder()
    df[col] = enc.fit_transform(df[col].astype(str))
    cat_encoders[col] = enc

feature_cols = [c for c in df.columns if c != 'NObeyesdad']
X = df[feature_cols].values.astype(float)
le = LabelEncoder()
y  = le.fit_transform(df['NObeyesdad'].values)
K  = len(le.classes_)
print(f"Kelas ({K}):", le.classes_.tolist())

scaler = StandardScaler()
X = scaler.fit_transform(X)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
def oh(y, k): m=np.zeros((len(y),k)); m[np.arange(len(y)),y]=1; return m
Ytr, Yte = oh(ytr, K), oh(yte, K)
print(f"Train: {Xtr.shape}, Test: {Xte.shape}")

relu   = lambda z: np.maximum(0, z)
drelu  = lambda z: (z > 0).astype(float)
def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
def loss_fn(yh, Y): return -np.mean(np.sum(Y * np.log(yh + 1e-9), 1))

n_features = Xtr.shape[1]
dims = [n_features, 64, 32, 16, K]
W = [np.random.randn(dims[i-1], dims[i]) * np.sqrt(2/dims[i-1]) for i in range(1, len(dims))]
b = [np.zeros((1, dims[i])) for i in range(1, len(dims))]

def forward(X, W, b):
    A, cache = X.copy(), []
    for i in range(len(W) - 1):
        Z = A @ W[i] + b[i]; cache.append((A, Z)); A = relu(Z)
    Z = A @ W[-1] + b[-1]; cache.append((A, Z))
    return softmax(Z), cache

def backward(yh, Y, cache, W):
    m = Y.shape[0]; gW=[None]*len(W); gb=[None]*len(W)
    dZ = (yh - Y)/m; Ap, _ = cache[-1]
    gW[-1] = Ap.T@dZ; gb[-1] = dZ.sum(0,keepdims=True); dA = dZ@W[-1].T
    for i in range(len(W)-2, -1, -1):
        Ap, Z = cache[i]; dZ = dA*drelu(Z)
        gW[i] = Ap.T@dZ; gb[i] = dZ.sum(0,keepdims=True); dA = dZ@W[i].T
    return gW, gb

mW=[np.zeros_like(w) for w in W]; vW=[np.zeros_like(w) for w in W]
mb_=[np.zeros_like(bi) for bi in b]; vb_=[np.zeros_like(bi) for bi in b]
def adam(W, b, gW, gb, t, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
    for i in range(len(W)):
        mW[i]=b1*mW[i]+(1-b1)*gW[i]; vW[i]=b2*vW[i]+(1-b2)*gW[i]**2
        mb_[i]=b1*mb_[i]+(1-b1)*gb[i]; vb_[i]=b2*vb_[i]+(1-b2)*gb[i]**2
        W[i]-=lr*(mW[i]/(1-b1**t))/(np.sqrt(vW[i]/(1-b2**t))+eps)
        b[i]-=lr*(mb_[i]/(1-b1**t))/(np.sqrt(vb_[i]/(1-b2**t))+eps)
    return W, b

EPOCHS=300; BS=64; PATIENCE=20; LR=0.001
best_vl=np.inf; wait=0; bW=[w.copy() for w in W]; bb=[bi.copy() for bi in b]
hist={"loss":[],"val_loss":[],"acc":[],"val_acc":[]}
print("="*55+"\n  Backpropagation Training — ObesityAI\n"+"="*55)

for ep in range(1, EPOCHS+1):
    idx=np.random.permutation(len(Xtr)); t=0
    for s in range(0, len(Xtr), BS):
        Xb=Xtr[idx[s:s+BS]]; Yb=Ytr[idx[s:s+BS]]; t+=1
        yh,c=forward(Xb,W,b); gW,gb=backward(yh,Yb,c,W); W,b=adam(W,b,gW,gb,t,LR)
    yh_tr,_=forward(Xtr,W,b); yh_va,_=forward(Xte,W,b)
    tl=loss_fn(yh_tr,Ytr); vl=loss_fn(yh_va,Yte)
    ta=accuracy_score(ytr,np.argmax(yh_tr,1)); va=accuracy_score(yte,np.argmax(yh_va,1))
    hist["loss"].append(tl); hist["val_loss"].append(vl)
    hist["acc"].append(ta);  hist["val_acc"].append(va)
    if ep%20==0 or ep==1:
        print(f"Epoch {ep:>3}  loss={tl:.4f}  val_loss={vl:.4f}  acc={ta:.4f}  val_acc={va:.4f}")
    if vl < best_vl-1e-5:
        best_vl=vl; wait=0; bW=[w.copy() for w in W]; bb=[bi.copy() for bi in b]
    else:
        wait+=1
        if wait>=PATIENCE:
            print(f"\nEarly Stopping di epoch {ep}"); W,b=bW,bb; break

ep_run=len(hist["loss"])
yh_f,_=forward(Xte,W,b); yp=np.argmax(yh_f,1)
acc=accuracy_score(yte,yp)
print(f"\n{'='*55}\n  Akurasi Akhir: {acc*100:.2f}%\n{'='*55}")
print(classification_report(yte,yp,target_names=le.classes_))

os.makedirs(os.path.join(BASE,"model"),exist_ok=True)
pickle.dump((W,b),        open(os.path.join(BASE,"model/weights.pkl"),"wb"))
pickle.dump(scaler,       open(os.path.join(BASE,"model/scaler.pkl"),"wb"))
pickle.dump(le,           open(os.path.join(BASE,"model/label_encoder.pkl"),"wb"))
pickle.dump(cat_encoders, open(os.path.join(BASE,"model/cat_encoders.pkl"),"wb"))
pickle.dump(dims,         open(os.path.join(BASE,"model/layer_dims.pkl"),"wb"))
pickle.dump(feature_cols, open(os.path.join(BASE,"model/feature_cols.pkl"),"wb"))
json.dump({"accuracy":round(acc*100,2),"epochs_run":ep_run,
    "final_loss":round(hist["loss"][-1],4),"final_val_loss":round(hist["val_loss"][-1],4),
    "classes":le.classes_.tolist(),"n_features":n_features,
    "train_size":len(Xtr),"test_size":len(Xte)},
    open(os.path.join(BASE,"model/metrics.json"),"w"))

os.makedirs(os.path.join(BASE,"static"),exist_ok=True)
fig,axs=plt.subplots(1,2,figsize=(12,5)); fig.patch.set_facecolor('#0f1117')
for ax in axs:
    ax.set_facecolor('#1a1d2e')
    for sp in ax.spines.values(): sp.set_color('#2d3748')
    ax.tick_params(colors='#a0aec0')
axs[0].plot(hist["loss"],color='#667eea',lw=2,label='Training Loss')
axs[0].plot(hist["val_loss"],color='#f6ad55',lw=2,label='Validation Loss')
axs[0].set_title('Loss per Epoch',color='white',fontsize=13,fontweight='bold')
axs[0].set_xlabel('Epoch',color='#a0aec0'); axs[0].set_ylabel('Loss',color='#a0aec0')
axs[0].legend(facecolor='#1a1d2e',labelcolor='white'); axs[0].grid(alpha=.2,color='#4a5568')
axs[1].plot(hist["acc"],color='#68d391',lw=2,label='Training Accuracy')
axs[1].plot(hist["val_acc"],color='#fc8181',lw=2,label='Validation Accuracy')
axs[1].set_title('Akurasi per Epoch',color='white',fontsize=13,fontweight='bold')
axs[1].set_xlabel('Epoch',color='#a0aec0'); axs[1].set_ylabel('Accuracy',color='#a0aec0')
axs[1].legend(facecolor='#1a1d2e',labelcolor='white'); axs[1].grid(alpha=.2,color='#4a5568')
plt.tight_layout()
plt.savefig(os.path.join(BASE,"static/training_history.png"),dpi=120,bbox_inches='tight',facecolor='#0f1117')
plt.close()
print("\n✅ Artefak & grafik disimpan!")
