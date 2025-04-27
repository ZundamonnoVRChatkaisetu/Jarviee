import React from 'react';
import { 
  AppBar, 
  Toolbar, 
  IconButton, 
  Typography, 
  Box, 
  Badge, 
  Menu, 
  MenuItem,
  ListItemIcon,
  ListItemText, 
  useTheme
} from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import NotificationsIcon from '@mui/icons-material/Notifications';
import SettingsIcon from '@mui/icons-material/Settings';
import AccountCircleIcon from '@mui/icons-material/AccountCircle';
import CodeIcon from '@mui/icons-material/Code';
import TuneIcon from '@mui/icons-material/Tune';
import NightsStayIcon from '@mui/icons-material/NightsStay';
import LightModeIcon from '@mui/icons-material/LightMode';
import ContrastIcon from '@mui/icons-material/Contrast';
import { useDispatch, useSelector } from 'react-redux';
import { toggleSidebar, setTheme } from '../../store/slices/uiSlice';
import { RootState } from '../../store';
import { motion } from 'framer-motion';
import { styled } from '@mui/material/styles';

// Styled components
const StyledAppBar = styled(AppBar)(({ theme }) => ({
  backdropFilter: 'blur(8px)',
  boxShadow: 'none',
  borderBottom: `1px solid ${theme.palette.divider}`,
  height: 'var(--header-height)',
  zIndex: theme.zIndex.drawer + 1,
}));

const LogoText = styled(Typography)(({ theme }) => ({
  fontWeight: 700,
  letterSpacing: 1,
  backgroundImage: `linear-gradient(45deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
  backgroundClip: 'text',
  WebkitBackgroundClip: 'text',
  color: 'transparent',
  textShadow: `0 0 10px ${theme.palette.primary.main}33`,
}));

const ArcReactorContainer = styled(motion.div)({
  width: 28,
  height: 28,
  borderRadius: '50%',
  position: 'relative',
  marginRight: 12,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
});

const ArcReactorOuter = styled('div')(({ theme }) => ({
  position: 'absolute',
  width: '100%',
  height: '100%',
  borderRadius: '50%',
  border: `2px solid ${theme.palette.primary.main}`,
}));

const ArcReactorInner = styled(motion.div)(({ theme }) => ({
  width: '60%',
  height: '60%',
  borderRadius: '50%',
  backgroundColor: theme.palette.primary.main,
}));

const Header: React.FC = () => {
  const theme = useTheme();
  const dispatch = useDispatch();
  const { theme: currentTheme } = useSelector((state: RootState) => state.ui);
  const { notifications } = useSelector((state: RootState) => state.ui);
  const unreadNotifications = notifications.filter(n => !n.read).length;
  
  const [userMenuAnchor, setUserMenuAnchor] = React.useState<null | HTMLElement>(null);
  const [notificationMenuAnchor, setNotificationMenuAnchor] = React.useState<null | HTMLElement>(null);
  const [themeMenuAnchor, setThemeMenuAnchor] = React.useState<null | HTMLElement>(null);
  
  const handleUserMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setUserMenuAnchor(event.currentTarget);
  };
  
  const handleUserMenuClose = () => {
    setUserMenuAnchor(null);
  };
  
  const handleNotificationMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setNotificationMenuAnchor(event.currentTarget);
  };
  
  const handleNotificationMenuClose = () => {
    setNotificationMenuAnchor(null);
  };
  
  const handleThemeMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setThemeMenuAnchor(event.currentTarget);
  };
  
  const handleThemeMenuClose = () => {
    setThemeMenuAnchor(null);
  };
  
  const handleThemeChange = (newTheme: 'jarvis' | 'light' | 'highContrast') => {
    dispatch(setTheme(newTheme));
    setThemeMenuAnchor(null);
  };
  
  return (
    <StyledAppBar position="fixed">
      <Toolbar variant="dense">
        <IconButton
          edge="start"
          color="inherit"
          aria-label="menu"
          onClick={() => dispatch(toggleSidebar())}
          sx={{ mr: 2 }}
        >
          <MenuIcon />
        </IconButton>
        
        <ArcReactorContainer
          initial={{ opacity: 0, scale: 0 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <ArcReactorOuter />
          <ArcReactorInner 
            animate={{ 
              boxShadow: ['0 0 5px #64B5F6', '0 0 15px #64B5F6', '0 0 5px #64B5F6'] 
            }}
            transition={{ 
              duration: 2, 
              repeat: Infinity, 
              ease: 'easeInOut' 
            }}
          />
        </ArcReactorContainer>
        
        <LogoText
          variant="h6"
          component={motion.h1}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5 }}
        >
          JARVIEE
        </LogoText>
        
        <Box sx={{ flexGrow: 1 }} />
        
        <Box sx={{ display: 'flex' }}>
          {/* Theme Switcher */}
          <IconButton 
            color="inherit" 
            onClick={handleThemeMenuOpen}
            aria-controls="theme-menu"
            aria-haspopup="true"
          >
            {currentTheme === 'jarvis' && <NightsStayIcon />}
            {currentTheme === 'light' && <LightModeIcon />}
            {currentTheme === 'highContrast' && <ContrastIcon />}
          </IconButton>
          <Menu
            id="theme-menu"
            anchorEl={themeMenuAnchor}
            open={Boolean(themeMenuAnchor)}
            onClose={handleThemeMenuClose}
            anchorOrigin={{
              vertical: 'bottom',
              horizontal: 'right',
            }}
            transformOrigin={{
              vertical: 'top',
              horizontal: 'right',
            }}
          >
            <MenuItem onClick={() => handleThemeChange('jarvis')}>
              <ListItemIcon>
                <NightsStayIcon fontSize="small" color={currentTheme === 'jarvis' ? 'primary' : 'inherit'} />
              </ListItemIcon>
              <ListItemText>JARVIS Dark</ListItemText>
            </MenuItem>
            <MenuItem onClick={() => handleThemeChange('light')}>
              <ListItemIcon>
                <LightModeIcon fontSize="small" color={currentTheme === 'light' ? 'primary' : 'inherit'} />
              </ListItemIcon>
              <ListItemText>Light</ListItemText>
            </MenuItem>
            <MenuItem onClick={() => handleThemeChange('highContrast')}>
              <ListItemIcon>
                <ContrastIcon fontSize="small" color={currentTheme === 'highContrast' ? 'primary' : 'inherit'} />
              </ListItemIcon>
              <ListItemText>High Contrast</ListItemText>
            </MenuItem>
          </Menu>
          
          {/* Notifications */}
          <IconButton 
            color="inherit"
            onClick={handleNotificationMenuOpen}
            aria-controls="notification-menu"
            aria-haspopup="true"
          >
            <Badge badgeContent={unreadNotifications} color="error">
              <NotificationsIcon />
            </Badge>
          </IconButton>
          <Menu
            id="notification-menu"
            anchorEl={notificationMenuAnchor}
            open={Boolean(notificationMenuAnchor)}
            onClose={handleNotificationMenuClose}
            anchorOrigin={{
              vertical: 'bottom',
              horizontal: 'right',
            }}
            transformOrigin={{
              vertical: 'top',
              horizontal: 'right',
            }}
            PaperProps={{
              sx: {
                width: 320,
                maxHeight: 400,
              },
            }}
          >
            {notifications.length === 0 ? (
              <MenuItem disabled>
                <ListItemText>No notifications</ListItemText>
              </MenuItem>
            ) : (
              notifications.slice(0, 5).map((notification) => (
                <MenuItem key={notification.id} onClick={handleNotificationMenuClose}>
                  <ListItemText
                    primary={notification.message}
                    secondary={new Date(notification.timestamp).toLocaleTimeString()}
                    primaryTypographyProps={{
                      variant: 'body2',
                      fontWeight: notification.read ? 'normal' : 'bold',
                    }}
                  />
                </MenuItem>
              ))
            )}
            {notifications.length > 5 && (
              <MenuItem onClick={handleNotificationMenuClose}>
                <ListItemText sx={{ textAlign: 'center' }}>View all notifications</ListItemText>
              </MenuItem>
            )}
          </Menu>
          
          {/* Settings */}
          <IconButton 
            color="inherit"
            component="a"
            href="/settings"
          >
            <SettingsIcon />
          </IconButton>
          
          {/* User Menu */}
          <IconButton
            edge="end"
            color="inherit"
            aria-label="account"
            aria-controls="user-menu"
            aria-haspopup="true"
            onClick={handleUserMenuOpen}
          >
            <AccountCircleIcon />
          </IconButton>
          <Menu
            id="user-menu"
            anchorEl={userMenuAnchor}
            open={Boolean(userMenuAnchor)}
            onClose={handleUserMenuClose}
            anchorOrigin={{
              vertical: 'bottom',
              horizontal: 'right',
            }}
            transformOrigin={{
              vertical: 'top',
              horizontal: 'right',
            }}
          >
            <MenuItem onClick={handleUserMenuClose}>
              <ListItemIcon>
                <AccountCircleIcon fontSize="small" />
              </ListItemIcon>
              <ListItemText>Profile</ListItemText>
            </MenuItem>
            <MenuItem onClick={handleUserMenuClose}>
              <ListItemIcon>
                <CodeIcon fontSize="small" />
              </ListItemIcon>
              <ListItemText>Developer Mode</ListItemText>
            </MenuItem>
            <MenuItem onClick={handleUserMenuClose}>
              <ListItemIcon>
                <TuneIcon fontSize="small" />
              </ListItemIcon>
              <ListItemText>Preferences</ListItemText>
            </MenuItem>
          </Menu>
        </Box>
      </Toolbar>
    </StyledAppBar>
  );
};

export default Header;
