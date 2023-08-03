from django.urls import path
from . import views


urlpatterns = [
    path('', views.index, name='index'),
    path('mask/', views.Face_Mask, name='mask'),

]