# Install/unInstall package files in LAMMPS
# mode = 0/1/2 for uninstall/install/update

mode=$1

# arg1 = file, arg2 = file it depends on

action () {
  if (test $mode = 0) then
    rm -f ../$1
  elif (! cmp -s $1 ../$1) then
    if (test -z "$2" || test -e ../$2) then
      ln -rs $1 ..
      if (test $mode = 2) then
        echo "  updating src/$1"
      fi
    fi
  elif (test -n "$2") then
    if (test ! -e ../$2) then
      rm -f ../$1
    fi
  fi
}

# force rebuild of files with LMP_USER_CUDA switch

touch ../accelerator_meso.h

# list of files with optional dependencies

for f in $(ls *.cu *.cpp *.h); do
    decu=${f/.cu/.cpp}
    demeso=${decu/_meso/}
    if (test -e ../${demeso}) then
        action $f ${demeso}
    else
        action $f
    fi
done

# edit 2 Makefile.package files to include/exclude package info

if (test $1 = 1) then

  if (test -e ../Makefile.package) then
    sed -i -e 's/[^ \t]*meso[^ \t]* //g' ../Makefile.package
    sed -i -e 's/[^ \t]*MESO[^ \t]* //g' ../Makefile.package
    sed -i -e 's|^PKG_INC =[ \t]*|&-DLMP_USER_MESO |' ../Makefile.package
    sed -i -e 's|^PKG_PATH =[ \t]*|&|' ../Makefile.package
    sed -i -e 's|^PKG_LIB =[ \t]*|&|' ../Makefile.package
    sed -i -e 's|^PKG_SYSINC =[ \t]*|&$(user-meso_SYSINC) |' ../Makefile.package
    sed -i -e 's|^PKG_SYSLIB =[ \t]*|&$(user-meso_SYSLIB) |' ../Makefile.package
    sed -i -e 's|^PKG_SYSPATH =[ \t]*|&$(user-meso_SYSPATH) |' ../Makefile.package
  fi

  if (test -e ../Makefile.package.settings) then
    sed -i -e '/^include.*meso.*$/d' ../Makefile.package.settings
  fi

elif (test $1 = 0) then

  if (test -e ../Makefile.package) then
    sed -i -e 's/[^ \t]*meso[^ \t]* //g' ../Makefile.package
    sed -i -e 's/[^ \t]*MESO[^ \t]* //g' ../Makefile.package
  fi

  if (test -e ../Makefile.package.settings) then
    sed -i -e '/^include.*meso.*$/d' ../Makefile.package.settings
  fi

fi

