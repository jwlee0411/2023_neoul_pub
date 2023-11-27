from django import forms
from .models import Post, Search


class PostForm(forms.ModelForm):
    class Meta:
        model = Post

        fields = ('title',)


class SearchForm(forms.ModelForm):
    class Meta:
        model = Search
        fields = ('search', 'type')

