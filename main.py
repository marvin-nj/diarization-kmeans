import sys
import numpy as np
import librosa 
from sklearn.cluster import KMeans
import audioMath
import audioFeature
import audioIo

WINSIZE=0.032  #32ms
is_feature_delta=True
is_feature_delat2=False
is_feature_norm=False
is_save=False

#ivectors + kmeans
def diarization3(filename, n_speakers,winSize=WINSIZE):
    print("---diarization3---")
    filename=filename.split('.')[0]
    matrix = np.loadtxt(open("ivectors/"+filename+".csv","rb"),skiprows=0)
    feature_all=matrix.reshape(1,-1)
    print(feature_all.shape)
    #kmeans
    kmeans = KMeans(n_clusters=n_speakers, init='k-means++',random_state=0)
    kmeans.fit(feature_all)
    cls=kmeans.labels_
    segs,flags = audioMath.labels_to_segments(cls, winSize)
    print("kmeans result:")
    for s in range(segs.shape[0]):
        print("{:.3f} {:.3f} {}".format(segs[s,0], segs[s,1], flags[s]))
    print("标签",len(cls),cls)
    #print("质心",kmeans.cluster_centers_)
    print("SSE",kmeans.inertia_)
    print("迭代次数",kmeans.n_iter_)
    print("分值",kmeans.score(feature_all.T))
 

#pyaudioAnalysis
def diarization2(filename, n_speakers,winSize=WINSIZE):
    #mfcc
    print("---diarization2---")
    sr,signal= audioIo.read_audio_file(filename)
    mid_window=1.28   #2.0s
    mid_step=0.08   #0.2
    short_window=0.032  #0.05s
    short_step=short_window*0.5

    feature_all,_,_=audioFeature.mid_feature_extraction(signal,sr,mid_window*sr, mid_step*sr,short_window*sr,short_step*sr)
    print("feature finally: ",feature_all.shape)

    #kmeans
    kmeans = KMeans(n_clusters=n_speakers, init='k-means++',random_state=0)
    kmeans.fit(feature_all.T)
    cls=kmeans.labels_
    segs,flags = audioMath.labels_to_segments(cls, mid_step)
    print("kmeans result:")
    for s in range(segs.shape[0]):
        print("{:.3f} {:.3f} {}".format(segs[s,0], segs[s,1], flags[s]))
    print("标签",len(cls),cls)
    #print("质心",kmeans.cluster_centers_)
    print("SSE",kmeans.inertia_)
    print("迭代次数",kmeans.n_iter_)
    print("分值",kmeans.score(feature_all.T))

#mfcc+kmeans
def diarization1(filename, n_speakers,winSize=WINSIZE):
    #mfcc
    y, sr = librosa.load(filename,sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    print(duration,sr)

    #feature mfcc
    n_win = int(winSize * sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, hop_length=n_win ,n_mfcc=24)
    print("mfcc ",mfccs.shape)
    if is_save:
        np.savetxt("feature.txt", mfccs , fmt='%f',delimiter=',')
    feature_all=mfccs
  
    #feature delta
    if is_feature_delta:
        delta = librosa.feature.delta(mfccs.T).T
        print("delta: ",delta.shape)
        #feature_all=np.vstack((mfccs,delta))
        feature_all=delta
    if is_feature_delat2:
        delta2 = librosa.feature.delta(mfccs,order=2)
        print("delta2: ",delta2.shape)
        feature_all=delta2

    #norm
    if is_feature_norm:
        feature_all_norm,mean,std=audioMath.normalize_features([feature_all.T])
        feature_all=feature_all_norm[0].T
        print("norm: ",feature_all.shape)
    
    print("feature finally: ",feature_all.shape)

    #kmeans
    kmeans = KMeans(n_clusters=n_speakers, init='k-means++',random_state=0)
    kmeans.fit(feature_all.T)
    cls=kmeans.labels_
    segs,flags = audioMath.labels_to_segments(cls, winSize)
    print("kmeans result:")
    for s in range(segs.shape[0]):
        print("{:.3f} {:.3f} {}".format(segs[s,0], segs[s,1], flags[s]))
    print("标签",len(cls),cls)
    #print("质心",kmeans.cluster_centers_)
    print("SSE",kmeans.inertia_)
    print("迭代次数",kmeans.n_iter_)
    print("分值",kmeans.score(feature_all.T))
    #y_hat = kmeans.predict(unknown)



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage : sys.argv[0] input.wav")
        sys.exit()
    wavf=sys.argv[1]
    nspeakers=3
    diarization1(wavf,nspeakers)
    diarization2(wavf,nspeakers)
    #diarization3(wavf,nspeakers)

