# Generated by Django 3.2.15 on 2023-01-25 13:24

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('common', '0005_alter_user_is_active'),
    ]

    operations = [
        migrations.AlterField(
            model_name='user',
            name='club_name',
            field=models.TextField(blank=True, max_length=30),
        ),
        migrations.AlterField(
            model_name='user',
            name='club_type',
            field=models.TextField(blank=True, max_length=30),
        ),
        migrations.AlterField(
            model_name='user',
            name='email',
            field=models.TextField(max_length=30),
        ),
    ]