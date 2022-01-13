from django.http import HttpResponse
from django.shortcuts import render

def index(request):
    params = {'name':'Aman','place':'Jab'}
    return render(request,'index.html',params)
def about(request):
    return HttpResponse("hello about")

    