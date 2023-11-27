import datetime

from django.db import models
from django.utils.crypto import get_random_string


class Search(models.Model):
    search = models.TextField(max_length=100)
    type = models.IntegerField(default=0)



class Post(models.Model):
    title = models.CharField(max_length=100)
    contents = models.TextField(max_length=5050)

    keyword1 = models.TextField(max_length=100, default="")
    keyword2 = models.TextField(max_length=100, default="")

    create_date = models.DateTimeField(auto_now_add=True)

    count = models.IntegerField(default=0)
    recommend = models.IntegerField(default=0)
