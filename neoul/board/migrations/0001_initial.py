# Generated by Django 3.2.15 on 2023-01-20 09:38

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Board',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('board_title', models.TextField(max_length=100, unique=True)),
                ('sub_id', models.IntegerField(unique=True)),
                ('is_superuser', models.BooleanField()),
                ('board_seq', models.IntegerField()),
            ],
        ),
        migrations.CreateModel(
            name='Post',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=100)),
                ('contents', models.TextField(max_length=5050)),
                ('create_date', models.DateTimeField(auto_now_add=True)),
                ('name', models.TextField(max_length=100)),
                ('board', models.CharField(choices=[('없음', '없음'), ('중요 공지사항', '중요 공지사항'), ('공지사항', '공지사항'), ('회의록', '회의록'), ('회칙', '회칙'), ('각종 서식', '각종 서식'), ('홍보', '홍보'), ('교양분과 활동보고서', '교양분과 활동보고서'), ('연대사업분과 활동보고서', '연대사업분과 활동보고서'), ('연행예술분과 활동보고서', '연행예술분과 활동보고서'), ('종교분과 활동보고서', '종교분과 활동보고서'), ('창작분과 활동보고서', '창작분과 활동보고서'), ('체육분과 활동보고서', '체육분과 활동보고서'), ('학술분과 활동보고서', '학술분과 활동보고서')], max_length=100)),
                ('file1', models.FileField(blank=True, null=True, upload_to='')),
                ('file2', models.FileField(blank=True, null=True, upload_to='')),
                ('file3', models.FileField(blank=True, null=True, upload_to='')),
                ('file4', models.FileField(blank=True, null=True, upload_to='')),
                ('file5', models.FileField(blank=True, null=True, upload_to='')),
            ],
        ),
    ]
