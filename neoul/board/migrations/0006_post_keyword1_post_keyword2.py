# Generated by Django 4.0 on 2023-11-19 19:56

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('board', '0005_post_recommend'),
    ]

    operations = [
        migrations.AddField(
            model_name='post',
            name='keyword1',
            field=models.TextField(default='', max_length=100),
        ),
        migrations.AddField(
            model_name='post',
            name='keyword2',
            field=models.TextField(default='', max_length=100),
        ),
    ]