import React from 'react';
import { 
  Drawer, 
  List, 
  ListItem, 
  ListItemButton, 
  ListItemIcon, 
  ListItemText, 
  Divider, 
  Box, 
  useTheme 
} from '@mui/material';
import { useNavigate, useLocation } from 'react-router-dom';
import { useSelector, useDispatch } from 'react-redux';
import { RootState } from '../../store';
import { setSidebarOpen } from '../../store/slices/uiSlice';
import DashboardIcon from '@mui/icons-material/Dashboard';
import CodeIcon from '@mui/icons-material/Code';
import StorageIcon from '@mui/icons-material/Storage';
import AssignmentIcon from '@mui/icons-material/Assignment';
import SettingsIcon from '@mui/icons-material/Settings';
import { styled } from '@mui/material/styles';
import { motion } from 'framer-motion';

// Styled components
const StyledDrawer = styled(Drawer)(({ theme }) => ({
  width: 'var(--sidebar-width)',
  flexShrink: 0,
  '& .MuiDrawer-paper': {
    width: 'var(--sidebar-width)',
    boxSizing: 'border-box',
    border: 'none',
    top: 'var(--header-height)',
    height: `calc(100% - var(--header-height))`,
    background: theme.palette.mode === 'dark' 
      ? 'rgba(33, 33, 33, 0.9)'
      : 'rgba(255, 255, 255, 0.9)',
    backdropFilter: 'blur(8px)',
  },
}));

const MenuItemButton = styled(ListItemButton)<{ active?: boolean }>(({ theme, active }) => ({
  borderRadius: 8,
  margin: '4px 8px',
  transition: theme.transitions.create(['background-color', 'box-shadow'], {
    duration: theme.transitions.duration.standard,
  }),
  '&:hover': {
    backgroundColor: theme.palette.mode === 'dark'
      ? 'rgba(255, 255, 255, 0.08)'
      : 'rgba(0, 0, 0, 0.04)',
  },
  ...(active && {
    backgroundColor: theme.palette.mode === 'dark'
      ? 'rgba(100, 181, 246, 0.2)'
      : 'rgba(25, 118, 210, 0.1)',
    boxShadow: theme.palette.mode === 'dark'
      ? `0 0 8px rgba(100, 181, 246, 0.3)`
      : 'none',
    '&:hover': {
      backgroundColor: theme.palette.mode === 'dark'
        ? 'rgba(100, 181, 246, 0.25)'
        : 'rgba(25, 118, 210, 0.15)',
    },
  }),
}));

// Navigation items
const navItems = [
  { 
    title: 'Dashboard', 
    icon: <DashboardIcon />, 
    path: '/dashboard',
    description: 'Main control center'
  },
  { 
    title: 'Code Editor', 
    icon: <CodeIcon />, 
    path: '/code-editor',
    description: 'Programming assistance'
  },
  { 
    title: 'Knowledge Explorer', 
    icon: <StorageIcon />, 
    path: '/knowledge-explorer',
    description: 'Browse knowledge base'
  },
  { 
    title: 'Task Manager', 
    icon: <AssignmentIcon />, 
    path: '/task-manager',
    description: 'Manage and monitor tasks'
  },
];

const bottomNavItems = [
  { 
    title: 'Settings', 
    icon: <SettingsIcon />, 
    path: '/settings',
    description: 'System configuration'
  },
];

const Sidebar: React.FC = () => {
  const theme = useTheme();
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const location = useLocation();
  const { sidebarOpen } = useSelector((state: RootState) => state.ui);
  
  const handleNavigate = (path: string) => {
    navigate(path);
    if (window.innerWidth < 768) {
      dispatch(setSidebarOpen(false));
    }
  };
  
  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
      },
    },
  };
  
  const item = {
    hidden: { opacity: 0, x: -20 },
    show: { opacity: 1, x: 0 },
  };
  
  return (
    <StyledDrawer
      variant="persistent"
      anchor="left"
      open={sidebarOpen}
    >
      <List component={motion.ul} variants={container} initial="hidden" animate="show">
        {navItems.map((item) => (
          <ListItem 
            key={item.title} 
            disablePadding 
            sx={{ my: 0.5 }} 
            component={motion.li}
            variants={item}
          >
            <MenuItemButton
              active={location.pathname === item.path}
              onClick={() => handleNavigate(item.path)}
            >
              <ListItemIcon sx={{ 
                color: location.pathname === item.path 
                  ? 'primary.main' 
                  : 'text.primary',
                minWidth: 40
              }}>
                {item.icon}
              </ListItemIcon>
              <ListItemText 
                primary={item.title} 
                secondary={item.description}
                primaryTypographyProps={{
                  variant: 'body2',
                  color: location.pathname === item.path ? 'primary' : 'inherit',
                }}
                secondaryTypographyProps={{
                  variant: 'caption',
                  sx: { 
                    opacity: 0.7,
                    transition: theme.transitions.create('opacity'),
                  }
                }}
              />
            </MenuItemButton>
          </ListItem>
        ))}
      </List>
      
      <Box sx={{ flexGrow: 1 }} />
      
      <Divider sx={{ mx: 2 }} />
      
      <List component={motion.ul} variants={container} initial="hidden" animate="show">
        {bottomNavItems.map((item) => (
          <ListItem 
            key={item.title} 
            disablePadding 
            sx={{ my: 0.5 }}
            component={motion.li}
            variants={item}
          >
            <MenuItemButton
              active={location.pathname === item.path}
              onClick={() => handleNavigate(item.path)}
            >
              <ListItemIcon sx={{ 
                color: location.pathname === item.path 
                  ? 'primary.main' 
                  : 'text.primary',
                minWidth: 40
              }}>
                {item.icon}
              </ListItemIcon>
              <ListItemText 
                primary={item.title} 
                secondary={item.description}
                primaryTypographyProps={{
                  variant: 'body2',
                  color: location.pathname === item.path ? 'primary' : 'inherit',
                }}
                secondaryTypographyProps={{
                  variant: 'caption',
                  sx: { opacity: 0.7 }
                }}
              />
            </MenuItemButton>
          </ListItem>
        ))}
      </List>
      
      <Box sx={{ p: 2, opacity: 0.7 }}>
        <Divider sx={{ mb: 1 }} />
        <Box sx={{ 
          fontSize: '0.75rem', 
          textAlign: 'center',
          color: 'text.secondary'
        }}>
          Jarviee v0.1.0
        </Box>
      </Box>
    </StyledDrawer>
  );
};

export default Sidebar;
