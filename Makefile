ifeq ($(OS),Windows_NT)     # is Windows_NT on XP, 2000, 7, Vista, 10...
    target := windows-gpu
else
    target := linux-gpu  # same as "uname -s"
endif

gpu: $(target)

windows-gpu:
	@powershell ./build.ps1

linux-gpu:
	./make.sh
