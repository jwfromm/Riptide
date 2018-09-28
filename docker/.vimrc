set nocompatible              " be iMproved, required
filetype off                  " required

" set the runtime path to include Vundle and initialize
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()

" let Vundle manage Vundle, required
Plugin 'VundleVim/Vundle.vim'
Plugin 'davidhalter/jedi-vim'

" All of your Plugins must be added before the following line
call vundle#end()            " required
filetype plugin indent on    " required

set nu
set expandtab
set tabstop=4
set shiftwidth=4
set hlsearch

syntax enable
set background=dark
"colorscheme solarized

set nocompatible
filetype off

" Find definition of current symbol using Gtags
map <C-?> <esc>:Gtags -r <CR>

" Find references to current symbol using Gtags
map <C-F> <esc>:Gtags <CR>

" Go to previous file
map <C-p> <esc>:bp<CR>

let g:jedi#popup_on_dot = 0
