# Generated by Django 4.0 on 2023-11-19 19:51

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('board', '0003_remove_post_file1_remove_post_file2_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='post',
            name='board',
        ),
    ]
