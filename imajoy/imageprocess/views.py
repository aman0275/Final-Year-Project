from http.client import HTTPResponse
from django.shortcuts import render
from django.http import HttpResponse
from .models import UploadImage
from imageprocess.forms import UserImage
from .process import run


# Create your views here.
def home(request):
    if request.method == 'POST':  
        form = UserImage(request.POST, request.FILES)  
        if form.is_valid():  
            form.save()  
            # Getting the current instance object to display in the template  
            img_object = form.instance  
            print("heheheheheh")
            print(img_object)
            print(type(img_object))
            run(img_object.image.url)
            # img_data = {
            #     "img1":"C:/Users/aman0/Downloads/processed image/1.jpg",
            #     "img2":"C:/Users/aman0/Downloads/processed image/2.jpg"
            # }
            return render(request, 'imageprocess/index.html', {'form': form, 'img_obj': img_object})  
    else:  
        form = UserImage()  
  

    return render(request,'imageprocess/index.html',{'form':form})


def test(request):
        return HttpResponse("testing")
