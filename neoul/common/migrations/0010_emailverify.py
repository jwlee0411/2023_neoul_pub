# Generated by Django 4.2 on 2023-04-26 23:04

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('common', '0009_user_email2'),
    ]

    operations = [
        migrations.CreateModel(
            name='EmailVerify',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('verify', models.TextField(max_length=10)),
            ],
        ),
    ]
