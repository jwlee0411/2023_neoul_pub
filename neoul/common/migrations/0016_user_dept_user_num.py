# Generated by Django 4.0 on 2023-06-04 19:27

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('common', '0015_clubinfo_image_01_clubinfo_image_02_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='user',
            name='dept',
            field=models.TextField(default=django.utils.timezone.now, max_length=30),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='user',
            name='num',
            field=models.TextField(default=django.utils.timezone.now, max_length=30),
            preserve_default=False,
        ),
    ]
