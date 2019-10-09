import collections
import math
import random
import zipfile
import numpy as np
import tensorflow as tf

def read_data_from_zipfile(filepath):#解压压缩文件
    with zipfile.ZipFile(filepath) as f:
        data=tf.compat.as_str(f.read(f.namelist()[0])).split()#通过tf.compat.as_str()#将数据转化为单词列表
    return data


#下面开始生成一个词汇表，我们使用collenctions.Counter()来计算单词出现的次数，并使用most_common来获取前50000个，使用python中的dict来建立
def build_dataset(words):
    count=[["UNK",-1]]
    word_count=collections.Counter(words)
    most_word=word_count.most_common(49999)
    #print(type(most_word))
    count.extend(most_word)
    word_dict=dict()
    
    
    for word,_ in count:
        word_dict[word]=len(word_dict)
    data=list()#记录编码
    unk_count=0
    for word in words:
        if word in word_dict:
            word_count=word_dict[word]
        else:
            word_count=0
            unk_count+=1
        data.append(word_count)
    count[0][1]=unk_count
    reverse_word_dict=dict(zip(word_dict.values(),word_dict.keys()))
    return data,count,word_dict,reverse_word_dict
                
words=read_data_from_zipfile("text8.zip")
encode,count,word_dict,reverse_word_dict=build_dataset(words)
#print(encode[:10],[reverse_word_dict[index] for index in encode[:10]])
del words

data_index=0
def generate_batch(batch_size,num_skips,skip_window):
    global data_index
    assert batch_size % num_skips==0
    assert num_skips <=2*skip_window
    
    x=np.ndarray(shape=(batch_size),dtype=np.int32)
    y=np.ndarray(shape=(batch_size,1),dtype=np.int32)
    
    deque_size=2*skip_window+1
    deque=collections.deque(maxlen=deque_size)
    
    #下面先初始化队列deque
    for _ in encode:
        deque.append(encode[data_index])
        data_index=(data_index+1)%len(encode)
        
    for i in range(batch_size // num_skips):
        target_word=skip_window
        avoid_word_target=[skip_window]
        for j in range(num_skips):
            while target_word in avoid_word_target:
                target_word=random.randint(0,deque_size-1)
            x[i*num_skips+j]=deque[skip_window]#采用skip-grams
            y[i*num_skips+j]=deque[target_word]
            avoid_word_target.append(target_word)
        deque.append(encode[data_index])
        data_index=(data_index+1)%len(encode)
    return x,y

# =============================================================================
# x,y=generate_batch(16,2,1)
# for i in range(16):
#     print(x[i],reverse_word_dict[x[i]],"---->",y[i,0],reverse_word_dict[y[i,0]])
#             
# =============================================================================
                

batch_size=128
embedding_szie=128
skip_window=1
num_skips=2

val_size=16
val_chiose=100

val_example=np.random.choice(val_chiose,val_size,replace=False)#从100个频数最多的单词中取16个观察他们是否具有很高的相关性
num_sample=64#负采样的数据个数



graph=tf.Graph()
with graph.as_default():
    #下面开始训练
    trainX=tf.placeholder(shape=[batch_size],dtype=tf.int32)
    trainY=tf.placeholder(shape=[batch_size,1],dtype=tf.int32)
    val_dataset=tf.constant(val_example,dtype=tf.int32)
    embeddings=tf.Variable(tf.random_uniform([50000,embedding_szie],-1.0,1.0))#初始化embedding矩阵，（50000,128）维向量
    embed=tf.nn.embedding_lookup(embeddings,trainX)#从embeddings中按照trainX为下标
    nce_weight=tf.Variable(tf.truncated_normal([50000,embedding_szie],stddev=1.0/math.sqrt(embedding_szie)))
    nce_bias=tf.Variable(tf.zeros([50000]))
    loss=tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,biases=nce_bias,inputs=embed,labels=trainY,num_sampled=num_sample,num_classes=50000))
    optimalizer=tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    
    
    #下面对训练好的embeding进行处理
    norm=tf.sqrt(tf.reduce_sum(tf.square(embeddings),1,keep_dims=True))
    norm_embedding=embeddings/norm#归一化训练好的embeddings
    val_embed=tf.nn.embedding_lookup(norm_embedding,val_dataset)
    
    similary=tf.matmul(norm_embedding,val_embed,transpose_b=True)
    init=tf.global_variables_initializer()
ecopes=100001
with tf.Session(graph=graph) as sess:
    init.run()
    print("初始化")
    average_loss=0
    for step in range(ecopes):
        x_input,y_input=generate_batch(batch_size,skip_window=skip_window,num_skips=num_skips)
        _,loss_val=sess.run([optimalizer,loss],feed_dict={trainX:x_input,trainY:y_input})
        average_loss+=loss_val
        
        print(step)
        if step %2000==0:#每2000次计算一个
            if step>0:
                average_loss/=2000
            print("average_loss : ",average_loss)
            average_loss=0
        if step %10000==0:#每10000次计算一次相似度
            sim=similary.eval()
            for i in range(val_size):
                valid_word=reverse_word_dict[val_example[i]]
                top_k=8
                
                nerast=(-sim[i,:]).argsort()[1:top_k+1]
                log_str="Nearest to %s:"%valid_word
                for k in range(top_k):
                    close_word=reverse_word_dict[nerast[k]]
                    log_str="%s %s,"%(log_str,close_word)
                print(log_str)
    final_embedding=norm_embedding.eval()
    
        
        
    




































    
    
