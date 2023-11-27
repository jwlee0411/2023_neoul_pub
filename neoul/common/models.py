from django.contrib.auth.models import AbstractUser
from django.db import models


# Create your models here.
class User(AbstractUser):
    email = models.TextField(max_length=30)
    email2 = models.TextField(max_length=30)
    phone_number = models.TextField(max_length=30)
    dept = models.TextField(max_length=30)
    num = models.TextField(max_length=30)
    first_name = models.TextField(max_length=30)
