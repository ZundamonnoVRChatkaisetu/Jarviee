import { createTheme, ThemeOptions } from '@mui/material/styles';

// Jarvisテーマ（デフォルト）
export const jarvisTheme: ThemeOptions = {
  palette: {
    mode: 'dark',
    primary: {
      main: '#64B5F6', // Arc Reactor Blue
      light: '#90CAF9',
      dark: '#42A5F5',
      contrastText: '#FFFFFF',
    },
    secondary: {
      main: '#4DD0E1', // Hologram Aquamarine
      light: '#80DEEA',
      dark: '#26C6DA',
      contrastText: '#000000',
    },
    error: {
      main: '#FF5252', // Accent Red
      light: '#FF8A80',
      dark: '#FF1744',
    },
    warning: {
      main: '#FFD54F', // Iron Gold
      light: '#FFE082',
      dark: '#FFCA28',
    },
    info: {
      main: '#2196F3',
      light: '#64B5F6',
      dark: '#1976D2',
    },
    success: {
      main: '#4CAF50',
      light: '#81C784',
      dark: '#388E3C',
    },
    background: {
      default: '#121212', // Dark Night Grey
      paper: 'rgba(66, 66, 66, 0.7)', // Asphalt Grey with transparency
    },
    text: {
      primary: '#FAFAFA', // Clear White
      secondary: 'rgba(255, 255, 255, 0.7)',
      disabled: 'rgba(255, 255, 255, 0.5)',
    },
    divider: 'rgba(255, 255, 255, 0.12)',
  },
  typography: {
    fontFamily: "'Roboto', -apple-system, BlinkMacSystemFont, 'Segoe UI', Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', sans-serif",
    h1: {
      fontSize: '2.5rem',
      fontWeight: 500,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 500,
    },
    h3: {
      fontSize: '1.75rem',
      fontWeight: 500,
    },
    h4: {
      fontSize: '1.5rem',
      fontWeight: 500,
    },
    h5: {
      fontSize: '1.25rem',
      fontWeight: 500,
    },
    h6: {
      fontSize: '1rem',
      fontWeight: 500,
    },
    body1: {
      fontSize: '1rem',
      lineHeight: 1.6,
    },
    body2: {
      fontSize: '0.875rem',
      lineHeight: 1.6,
    },
    button: {
      fontWeight: 500,
      textTransform: 'none',
    },
    caption: {
      fontSize: '0.75rem',
    },
  },
  components: {
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: 'rgba(18, 18, 18, 0.8)',
          backdropFilter: 'blur(8px)',
          boxShadow: '0 4px 8px rgba(0, 0, 0, 0.3)',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          textTransform: 'none',
          padding: '8px 16px',
        },
        contained: {
          boxShadow: '0 4px 8px rgba(0, 0, 0, 0.2)',
          '&:hover': {
            boxShadow: '0 6px 12px rgba(0, 0, 0, 0.3)',
          },
        },
        outlined: {
          borderWidth: 2,
          '&:hover': {
            borderWidth: 2,
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundColor: 'rgba(66, 66, 66, 0.7)',
          backdropFilter: 'blur(8px)',
          borderRadius: 12,
          boxShadow: '0 8px 16px rgba(0, 0, 0, 0.3)',
          border: '1px solid rgba(255, 255, 255, 0.12)',
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
        },
        elevation1: {
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.15)',
        },
        elevation2: {
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.2)',
        },
        elevation3: {
          boxShadow: '0 6px 16px rgba(0, 0, 0, 0.25)',
        },
        elevation4: {
          boxShadow: '0 8px 20px rgba(0, 0, 0, 0.3)',
        },
      },
    },
    MuiDrawer: {
      styleOverrides: {
        paper: {
          backgroundColor: 'rgba(33, 33, 33, 0.9)',
          backdropFilter: 'blur(8px)',
          borderRight: '1px solid rgba(255, 255, 255, 0.12)',
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 8,
          },
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 8,
        },
      },
    },
    MuiTooltip: {
      styleOverrides: {
        tooltip: {
          backgroundColor: 'rgba(77, 208, 225, 0.9)',
          color: '#000000',
          fontSize: '0.75rem',
          borderRadius: 4,
          backdropFilter: 'blur(4px)',
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.2)',
        },
        arrow: {
          color: 'rgba(77, 208, 225, 0.9)',
        },
      },
    },
    MuiCssBaseline: {
      styleOverrides: {
        ':root': {
          '--arc-reactor-blue': '#64B5F6',
          '--hologram-aquamarine': '#4DD0E1',
          '--accent-red': '#FF5252',
          '--iron-gold': '#FFD54F',
          '--dark-night-grey': '#121212',
          '--asphalt-grey': '#424242',
          '--clear-white': '#FAFAFA',
        },
        body: {
          scrollbarWidth: 'thin',
          scrollbarColor: '#64B5F6 #424242',
          '&::-webkit-scrollbar': {
            width: '8px',
            height: '8px',
          },
          '&::-webkit-scrollbar-track': {
            background: '#424242',
          },
          '&::-webkit-scrollbar-thumb': {
            backgroundColor: '#64B5F6',
            borderRadius: '4px',
          },
        },
      },
    },
  },
  shape: {
    borderRadius: 8,
  },
  transitions: {
    duration: {
      shortest: 150,
      shorter: 200,
      short: 250,
      standard: 300,
      complex: 375,
      enteringScreen: 225,
      leavingScreen: 195,
    },
  },
  zIndex: {
    mobileStepper: 1000,
    speedDial: 1050,
    appBar: 1100,
    drawer: 1200,
    modal: 1300,
    snackbar: 1400,
    tooltip: 1500,
  },
};

// ライトテーマ
export const lightTheme: ThemeOptions = {
  ...jarvisTheme,
  palette: {
    ...jarvisTheme.palette,
    mode: 'light',
    primary: {
      main: '#1976D2',
      light: '#42A5F5',
      dark: '#0D47A1',
      contrastText: '#FFFFFF',
    },
    secondary: {
      main: '#0097A7',
      light: '#26C6DA',
      dark: '#00838F',
      contrastText: '#FFFFFF',
    },
    background: {
      default: '#FFFFFF',
      paper: '#F5F5F5',
    },
    text: {
      primary: '#212121',
      secondary: '#757575',
      disabled: '#9E9E9E',
    },
    divider: 'rgba(0, 0, 0, 0.12)',
  },
  components: {
    ...jarvisTheme.components,
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: 'rgba(255, 255, 255, 0.8)',
          backdropFilter: 'blur(8px)',
          boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)',
          color: '#212121',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundColor: '#FFFFFF',
          backdropFilter: 'none',
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
          border: '1px solid rgba(0, 0, 0, 0.08)',
        },
      },
    },
    MuiDrawer: {
      styleOverrides: {
        paper: {
          backgroundColor: '#FFFFFF',
          backdropFilter: 'none',
          borderRight: '1px solid rgba(0, 0, 0, 0.12)',
        },
      },
    },
    MuiTooltip: {
      styleOverrides: {
        tooltip: {
          backgroundColor: 'rgba(25, 118, 210, 0.9)',
          color: '#FFFFFF',
        },
        arrow: {
          color: 'rgba(25, 118, 210, 0.9)',
        },
      },
    },
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          scrollbarColor: '#1976D2 #E0E0E0',
          '&::-webkit-scrollbar-track': {
            background: '#E0E0E0',
          },
          '&::-webkit-scrollbar-thumb': {
            backgroundColor: '#1976D2',
          },
        },
      },
    },
  },
};

// 高コントラストテーマ
export const highContrastTheme: ThemeOptions = {
  ...jarvisTheme,
  palette: {
    ...jarvisTheme.palette,
    mode: 'dark',
    primary: {
      main: '#00B0FF',
      light: '#40C4FF',
      dark: '#0091EA',
      contrastText: '#FFFFFF',
    },
    secondary: {
      main: '#00E5FF',
      light: '#18FFFF',
      dark: '#00B8D4',
      contrastText: '#000000',
    },
    error: {
      main: '#FF1744',
      light: '#FF4081',
      dark: '#D50000',
    },
    warning: {
      main: '#FFEA00',
      light: '#FFFF00',
      dark: '#FFD600',
    },
    background: {
      default: '#000000',
      paper: '#121212',
    },
    text: {
      primary: '#FFFFFF',
      secondary: '#EEEEEE',
      disabled: '#BDBDBD',
    },
    divider: 'rgba(255, 255, 255, 0.3)',
  },
  components: {
    ...jarvisTheme.components,
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 4,
          padding: '10px 20px',
          fontWeight: 700,
        },
        contained: {
          boxShadow: 'none',
          '&:hover': {
            boxShadow: 'none',
          },
        },
        outlined: {
          borderWidth: 2,
          '&:hover': {
            borderWidth: 2,
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundColor: '#121212',
          backdropFilter: 'none',
          boxShadow: '0 0 0 2px #FFFFFF',
          borderRadius: 4,
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
        },
        elevation1: {
          boxShadow: '0 0 0 1px #FFFFFF',
        },
        elevation2: {
          boxShadow: '0 0 0 2px #FFFFFF',
        },
        elevation3: {
          boxShadow: '0 0 0 3px #FFFFFF',
        },
        elevation4: {
          boxShadow: '0 0 0 4px #FFFFFF',
        },
      },
    },
    MuiDrawer: {
      styleOverrides: {
        paper: {
          backgroundColor: '#000000',
          backdropFilter: 'none',
          borderRight: '1px solid #FFFFFF',
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: '#000000',
          backdropFilter: 'none',
          boxShadow: '0 1px 0 #FFFFFF',
        },
      },
    },
    MuiTooltip: {
      styleOverrides: {
        tooltip: {
          backgroundColor: '#FFFFFF',
          color: '#000000',
          fontWeight: 700,
          fontSize: '0.875rem',
        },
        arrow: {
          color: '#FFFFFF',
        },
      },
    },
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          scrollbarColor: '#FFFFFF #000000',
          '&::-webkit-scrollbar-track': {
            background: '#000000',
          },
          '&::-webkit-scrollbar-thumb': {
            backgroundColor: '#FFFFFF',
          },
        },
      },
    },
  },
  shape: {
    borderRadius: 4,
  },
};

// テーマの作成と取得
export const getTheme = (themeName: 'jarvis' | 'light' | 'highContrast' | 'custom' = 'jarvis', customOptions?: ThemeOptions) => {
  let themeOptions: ThemeOptions;
  
  switch (themeName) {
    case 'light':
      themeOptions = lightTheme;
      break;
    case 'highContrast':
      themeOptions = highContrastTheme;
      break;
    case 'custom':
      themeOptions = customOptions || jarvisTheme;
      break;
    case 'jarvis':
    default:
      themeOptions = jarvisTheme;
      break;
  }
  
  return createTheme(themeOptions);
};

export default getTheme;
