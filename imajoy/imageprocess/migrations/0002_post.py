# Generated by Django 4.0.1 on 2022-06-04 09:18

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('imageprocess', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Post',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('post_image', models.ImageField(upload_to='images/')),
                ('post_date', models.DateField(auto_now_add=True)),
            ],
        ),
    ]
