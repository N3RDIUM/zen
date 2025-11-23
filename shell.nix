let pkgs = import <nixpkgs> { };
in pkgs.mkShell {
    packages = [
        (pkgs.python3.withPackages (python-pkgs: [
            python-pkgs.torch
            python-pkgs.ollama
            python-pkgs.faster-whisper
            python-pkgs.speechrecognition
            python-pkgs.transformers
            python-pkgs.scipy
            python-pkgs.pydub
            python-pkgs.pyaudio
            python-pkgs.onnxruntime
            python-pkgs.soundfile
        ]))
        pkgs.pyright
    ];
    buildInputs = [ pkgs.libz ];
}
