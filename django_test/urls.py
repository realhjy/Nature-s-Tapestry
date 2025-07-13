"""django_test URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from homepage import views as homepage_views

urlpatterns = [
    path('', homepage_views.homepage),
    path('imageList', homepage_views.image_list_page),
    path('videoList', homepage_views.video_list_page),
    path('upload', homepage_views.upload),
    path('upload_image', homepage_views.upload_image),
    path('upload_video', homepage_views.upload_video),
    path('search_image', homepage_views.search_image),
    path('search_video', homepage_views.search_video),
    path('imageProcessing', homepage_views.imageProcessing),
    path('load_images', homepage_views.load_images),
    path('grayscale_image', homepage_views.grayscale_image),
    path('equalize_image', homepage_views.equalize_image),
    path('audio_play', homepage_views.audio_play),
    path('upload_error/<str:upload_error>', homepage_views.upload_error, name='upload_error'),
    path('ai_assistant', homepage_views.ai_assistant),
    path('api/ask-baidu/', homepage_views.ask_baidu_api, name='ask_baidu_api'),
    path('api/ask-baidu-stream/', homepage_views.ask_streaming_api, name='ask_baidu_streaming_api'),
    path('ai_recognize_image/', homepage_views.ai_recognize_image, name='ai_recognize_image'),
    path('ai_analyze_video/', homepage_views.ai_analyze_video, name='ai_analyze_video'),
]

