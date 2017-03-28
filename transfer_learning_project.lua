require 'nn'
require 'image'
require 'optim'

logger = optim.Logger('Transfer.log') -- logger can be changed  
logger:setNames{'Trainset Error', 'Testset Error'}

NUM_CLASSES = 4
dataset = torch.load('flowers.t7')
dataset = dataset:narrow(1,1,NUM_CLASSES)
classes = torch.range(1,NUM_CLASSES):totable() --17 classes
labels = torch.range(1,NUM_CLASSES):view(NUM_CLASSES,1):expand(NUM_CLASSES,80)

print(dataset:size()) --each class has 80 images of 3x128x128
image.display(dataset:select(2,20))

function shuffle(data,ydata) --shuffle data function
    local RandOrder = torch.randperm(data:size(1)):long()
    return data:index(1,RandOrder), ydata:index(1,RandOrder)
end



shuffledData, shuffledLabels = shuffle(dataset:view(-1,3,128,128), labels:contiguous():view(-1))

trainSize = 0.85 * shuffledData:size(1)
trainData, testData = unpack(shuffledData:split(trainSize, 1))
trainLabels, testLabels = unpack(shuffledLabels:split(trainSize, 1))

print(trainData:size())

trainData = trainData:float() -- convert the data from a ByteTensor to a float Tensor.
trainLabels = trainLabels:float()

mean, std = trainData:mean(), trainData:std()
print(mean, std)
trainData:add(-mean):div(std)
    
testData = testData:float()
testLabels = testLabels:float()
testData:add(-mean):div(std)


-- Load GoogLeNet
googLeNet = torch.load('GoogLeNet_v2_nn.t7')

-- The new network
model = nn.Sequential()

for i=1,10 do
    local layer = googLeNet:get(i):clone()
    layer.parameters = function() return {} end --disable parameters
    layer.accGradParamters = nil --remove accGradParamters
    model:add(layer)
end

-- Check output dimensions with random input
model:float()
local y = model:forward(torch.rand(1,3,128,128):float())
print(y:size())

-- Add the new layers

viewSize = 16 * 3 * 3
model:add(nn.SpatialConvolution(320, 16, 3, 3)) -- Input: 320, Output: 16, Kernel: 3X3, Stride: 1, Padding: 0
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(4, 4, 4, 4)) -- Kernel: 4X4, Stride: 4
model:add(nn.View(viewSize)) -- View: Features * height * width (features: 16 - from SpatialConvolution, Width & Height: 128 / 4 - SpatialMaxPooling stride is 4)
model:add(nn.Dropout(0.5)) -- Dropout layer with 50% probability
model:add(nn.Linear(viewSize, NUM_CLASSES))
model:add(nn.LogSoftMax())

model:float()

-- Loss Function = Negative Log Likelihood ()
lossFunc = nn.ClassNLLCriterion():float()
w, dE_dw = model:getParameters()
 
print('Number of parameters:', w:nElement())
 
batchSize = 32
epochs = 200
optimState = {
    learningRate = 0.1,
}

function forwardNet(data, labels, train)

    --another helpful function of optim is ConfusionMatrix
    local confusion = optim.ConfusionMatrix(torch.range(0,NUM_CLASSES - 1):totable())
    local lossAcc = 0
    local numBatches = 0
	
    if train then
        --set network into training mode
        model:training()
    end
	
	
    for i = 1, data:size(1)-batchSize, batchSize do
        numBatches = numBatches + 1
        local x = data:narrow(1, i, batchSize):float()
        local yt = labels:narrow(1, i, batchSize):float()
        local y = model:forward(x)
        local err = lossFunc:forward(y, yt)
        lossAcc = lossAcc + err
		
        confusion:batchAdd(y,yt)
       
        if train then
            function feval()
                --model:zeroGradParameters() --zero grads
                local dE_dy = lossFunc:backward(y,yt)
                model:backward(x, dE_dy) -- backpropagation
           
                return err, dE_dw
            end
           
            optim.adam(feval, w, optimState)
        end
    end
   
    confusion:updateValids()
    local avgLoss = lossAcc / numBatches
    local avgError = 1 - confusion.totalValid
	
	--print("train=========")
	--print(train)
	--print("train=========")
	--print("confusion==============================")
	--print(confusion)
	--print("confusion==============================")
   
    return avgLoss, avgError, tostring(confusion)
end
 
trainLoss = torch.Tensor(epochs)
testLoss = torch.Tensor(epochs)
trainError = torch.Tensor(epochs)
testError = torch.Tensor(epochs)
 
--reset net weights
model:apply(function(l) l:reset() end)
 
for e = 1, epochs do
     trainData, trainLabels = shuffle(trainData, trainLabels) --shuffle training data
    trainLoss[e], trainError[e] = forwardNet(trainData, trainLabels, true)
    testLoss[e], testError[e], confusion = forwardNet(testData, testLabels, false)
    logger:add{trainError[e],testError[e]} -- loss is the value which you want to plot
    logger:style{'-','-'}   -- the style of your line, as in MATLAB, we use '-' or '|' etc.
 
    if e % 5 == 0 then
        print('Epoch ' .. e .. ':')
        print('Training error: ' .. trainError[e], 'Training Loss: ' .. trainLoss[e])
        print('Test error: ' .. testError[e], 'Test Loss: ' .. testLoss[e])
        print(confusion)
 
    end
end
 
logger:plot()