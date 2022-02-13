#!/usr/bin/env bash
# execute `sudo pacman -S qt5-xmlpatterns` first
MITSUBA_DIR=$(dirname "$BASH_SOURCE")
cd $MITSUBA_DIR
ln -s build/config-linux-gcc.py config.py
cd /usr/include
sudo ln -s qt qt5
cd /usr/lib/pkgconfig
sudo ln -s Qt5Core.pc QtCore.pc
sudo ln -s Qt5Gui.pc QtGui.pc
sudo ln -s Qt5Network.pc QtNetwork.pc
sudo ln -s Qt5OpenGL.pc QtOpenGL.pc
sudo ln -s Qt5Widgets.pc QtWidgets.pc
sudo ln -s Qt5Xml.pc QtXml.pc
sudo ln -s Qt5XmlPatterns.pc QtXmlPatterns.pc