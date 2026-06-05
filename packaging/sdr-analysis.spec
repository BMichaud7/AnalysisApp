%global debug_package %{nil}
Name:       sdr-analysis
Version:    %{pkg_version}
Release:    1%{?dist}
Summary:    OpenRFStack ONNX-based automatic modulation recognition
License:    Proprietary
URL:        https://github.com/OpenRFStack/AnalysisApp
BuildArch:  x86_64
AutoReqProv: no
Requires:   qpid-proton-cpp tinyxml2 fftw spdlog fmt

%description
AnalysisApp classifies RF signals using a 47-class RadioResNet ONNX model.
Runs GPU inference (CUDA via ONNX Runtime) on 1024-sample IQ snapshots
from rf.detections messages. Publishes modulation type and confidence back
to AMQP. Supports CUDA and CPU inference modes.

%prep
%build
%install

%files
/usr/bin/sdr_analysis

%changelog
* Thu Jan 01 2026 OpenRFStack CI <noreply@github.com> - %{pkg_version}-1
- Automated build from main/1.0
