import numpy as np
from PIL import Image
import face.transforms as transforms
from skimage import io
from skimage.transform import resize
from face.models import *
from process  import *
cut_size = 44

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

def dele(arr,index):
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return torch.cat((arr1,arr2),dim=0)
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

for imgs in range (len(os.listdir('face/set'))):
    imgpath=os.path.join('face/set/%s_1.jpg'%imgs)
    raw_img = io.imread(imgpath)
    gray = rgb2gray(raw_img)
    gray = resize(gray, (48,48), mode='symmetric').astype(np.uint8)

    img = gray[:, :, np.newaxis]

    img = np.concatenate((img, img, img), axis=2)
    img = Image.fromarray(img)
    inputs = transform_test(img)

    class_names = [ 'Happy', 'Sad',  'Neutral']

    net = VGG('VGG19')
    checkpoint = torch.load(os.path.join('face/FER2013_VGG19', 'PrivateTest_model.t7'),map_location='cpu')
    net.load_state_dict(checkpoint['net'])
    #net.cuda()
    net.eval()

    ncrops, c, h, w = np.shape(inputs)

    inputs = inputs.view(-1, c, h, w)
    #inputs = inputs.cuda()
    inputs = Variable(inputs, volatile=True)
    outputs = net(inputs)

    outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops
    outputs_avg=outputs_avg[3:]
    outputs_avg=dele(outputs_avg,2)

    score = F.softmax(outputs_avg)
    print(score)
    _, predicted = torch.max(outputs_avg.data, 0)
    print("表情为： %s" %str(class_names[int(predicted.cpu().numpy())]))



