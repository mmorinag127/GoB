

# install tmux
## livevent
```sh
cd local/src
curl -LO https://github.com/libevent/libevent/releases/download/release-2.1.11-stable/libevent-2.1.11-stable.tar.gz
tar xf libevent-2.1.11-stable.tar.gz
cd libevent-2.1.11-stable
./configure --prefix=$HOME/local --disable-shared --disable-libevent-regress
make && make verify && make install
```

## ncurses
```sh
cd local/src
curl -LO https://invisible-mirror.net/archives/ncurses/ncurses-6.1.tar.gz
tar xf ncurses-6.1.tar.gz
cd ncurses-6.1
./configure --prefix=$HOME/local --enable-pc-files --with-pkg-config=$HOME/local/lib/pkgconfig
make && make install
```

## update .zprofile

```sh
echo "export PKG_CONFIG_PATH=$HOME/local/lib/pkgconfig:$PKG_CONFIG_PATH" >> ~/.zprofile
```

## tmux

```sh
git clone https://github.com/tmux/tmux
cd tmux
./autogen.sh
./configure --prefix=/home/morinaga/local/
make && make install
```


## ncurses-5.9
```
wget https://ftp.gnu.org/pub/gnu/ncurses/ncurses-5.9.tar.gz
tar xf ncurses-5.9.tar.gz
cd ncurses-5.9
/configure --prefix=$HOME/local --enable-pc-files --with-pkg-config=$HOME/local/lib/pkgconfig
make && make install
```
