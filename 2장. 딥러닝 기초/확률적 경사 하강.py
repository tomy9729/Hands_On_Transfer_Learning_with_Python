import numpy as np
#확률적 경사 하강
#계산대에서의 진짜 가격
p_fish = 150; p_ketchup = 50; p_chips = 100
#식사 가격 샘플 : 10일 동안의 식사 가격을 일반화한 데이터
np.random.seed(100)
portions = np.random.randint(low=1,high=10,size=3)
X=[]; y=[]; days=10
for i in range(days):
  portions = np.random.randint(low=1,high=10,size=3)
  price = p_fish * portions[0] + p_ketchup * portions[1] + p_chips * portions[2]
  X.append(portions)
  y.append(price)

X = np.array(X)
y = np.array(y)

#선형 모델 만들기
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD

price_guess = [np.array([[ 50 ],[ 50],[ 50 ]])]#가격에 대한 초기 추측
model_input = Input(shape=(3,),dtype='float32')
model_output = Dense(1,activation = 'linear', use_bias = False, 
                     name = 'LinearNeuron', 
                     weights=price_guess)(model_input)
sgd=SGD(lr=0.01)
model=Model(model_input, model_output)

#제곱 오차 손실 E의 확률적 경사 하강도(SGD)) 정의
#optimizer
model.compile(loss='mean_squared_error',optimizer=sgd)
model.summary()

#반복 최적화에 의한 훈련 모델 : 5사이즈의 미니 배치 SGD
history = model.fit(X,y,batch_size = 5, epochs=30,verbose=2)
