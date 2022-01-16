from django.http import HttpResponse
from django.shortcuts import render

def index(request):
    params = {'name':'Aman','place':'Jab'}
    return HttpResponse("hello about")
def about(request):
    djtext = request.GET.get('text','default')
    # for parameter passing through form
    return HttpResponse("hello about")

    