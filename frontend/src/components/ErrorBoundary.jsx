import { Component } from "react";

// Oq (bo'sh) sahifa xatosi tuzatishi: avval DialogSessionView daraxtida
// (yoki uning ichidagi useDialogSession.js kabi hook'larda) sodir bo'lgan
// HAR QANDAY kutilmagan render/effekt xatosi React'ni butun daraxtni
// o'chirib, HECH NARSA ko'rsatmasdan qoldirardi — chunki hech qanday
// ErrorBoundary yo'q edi. React error boundary'lari FAQAT klass
// komponent sifatida yozilishi SHART (hook ekvivalenti yo'q).
export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    console.error("[ErrorBoundary] ushlangan xato:", error, info?.componentStack);
  }

  handleRetry = () => {
    this.setState({ error: null });
    this.props.onReset?.();
  };

  render() {
    if (this.state.error) {
      return (
        <div className="min-h-screen flex flex-col items-center justify-center gap-6 px-6">
          <p className="text-red-500 text-lg text-center max-w-md">
            Kutilmagan xatolik yuz berdi.
          </p>
          <p className="text-gray-400 text-sm text-center max-w-md">
            {this.state.error?.message || String(this.state.error)}
          </p>
          <button
            onClick={this.handleRetry}
            className="px-8 py-4 rounded-full bg-purple-400 shadow-md text-white font-medium hover:scale-105 transition-transform"
          >
            Qayta urinish
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
