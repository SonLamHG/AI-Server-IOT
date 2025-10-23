from django.urls import path

from . import views


urlpatterns = [
    path('', views.index, name='video_index'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('snapshot/', views.snapshot, name='snapshot'),
    path('fire_list/', views.fire_list, name='fire_list'),
    path('fire_events/', views.fire_events, name='fire_events'),
    path('analysis/', views.simple_analysis, name='simple_analysis'),
]
