from pathlib import Path
from typing import Optional, Dict, Any
import weakref
from PyQt6.QtGui import QPixmap, QIcon
from PyQt6.QtCore import QUrl
from .logger import logger

class ResourceManager:
    """Manages application resources with caching and efficient loading."""
    
    def __init__(self):
        # Use weak references for cache to allow garbage collection
        self._pixmap_cache: Dict[str, weakref.ref] = {}
        self._icon_cache: Dict[str, weakref.ref] = {}
        self._url_cache: Dict[str, QUrl] = {}
        
        # Resource paths
        self._resource_paths = {
            'video': Path('DySCO_opener.mp4'),
            'music': Path('The_Tramps_Disco_Inferno.mp3'),
            'disco_ball': Path('Disco_ball_blackBG.jpeg')
        }
    
    def get_resource_path(self, resource_name: str) -> Optional[Path]:
        """Get the path for a named resource."""
        if resource_name not in self._resource_paths:
            logger.warning(f"Unknown resource: {resource_name}")
            return None
        return self._resource_paths[resource_name]
    
    def get_pixmap(self, resource_name: str, size: Optional[tuple] = None) -> Optional[QPixmap]:
        """
        Get a QPixmap from cache or load it from disk.
        
        Args:
            resource_name: Name of the resource
            size: Optional (width, height) tuple for scaling
        """
        # Check cache first
        if resource_name in self._pixmap_cache:
            pixmap = self._pixmap_cache[resource_name]()
            if pixmap is not None:
                logger.debug(f"Retrieved {resource_name} from pixmap cache")
                return pixmap
        
        # Load from disk
        path = self.get_resource_path(resource_name)
        if path is None:
            return None
            
        try:
            pixmap = QPixmap(str(path))
            if not pixmap.isNull():
                if size:
                    pixmap = pixmap.scaled(
                        size[0], size[1],
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                # Cache the pixmap
                self._pixmap_cache[resource_name] = weakref.ref(pixmap)
                logger.debug(f"Cached {resource_name} pixmap")
                return pixmap
            else:
                logger.error(f"Failed to load pixmap: {path}")
                return None
        except Exception as e:
            logger.error(f"Error loading pixmap {path}: {str(e)}")
            return None
    
    def get_icon(self, resource_name: str, size: Optional[tuple] = None) -> Optional[QIcon]:
        """
        Get a QIcon from cache or create it from a pixmap.
        
        Args:
            resource_name: Name of the resource
            size: Optional (width, height) tuple for scaling
        """
        # Check cache first
        if resource_name in self._icon_cache:
            icon = self._icon_cache[resource_name]()
            if icon is not None:
                logger.debug(f"Retrieved {resource_name} from icon cache")
                return icon
        
        # Create from pixmap
        pixmap = self.get_pixmap(resource_name, size)
        if pixmap is not None:
            icon = QIcon(pixmap)
            # Cache the icon
            self._icon_cache[resource_name] = weakref.ref(icon)
            logger.debug(f"Cached {resource_name} icon")
            return icon
        return None
    
    def get_url(self, resource_name: str) -> Optional[QUrl]:
        """Get a QUrl for a resource, using cache if available."""
        # Check cache first
        if resource_name in self._url_cache:
            logger.debug(f"Retrieved {resource_name} from URL cache")
            return self._url_cache[resource_name]
        
        # Create new URL
        path = self.get_resource_path(resource_name)
        if path is None:
            return None
            
        try:
            url = QUrl.fromLocalFile(str(path))
            # Cache the URL
            self._url_cache[resource_name] = url
            logger.debug(f"Cached {resource_name} URL")
            return url
        except Exception as e:
            logger.error(f"Error creating URL for {path}: {str(e)}")
            return None
    
    def clear_cache(self):
        """Clear all caches."""
        self._pixmap_cache.clear()
        self._icon_cache.clear()
        self._url_cache.clear()
        logger.info("Resource caches cleared")
    
    def preload_resources(self):
        """Preload commonly used resources."""
        logger.info("Preloading resources...")
        for resource in self._resource_paths:
            self.get_url(resource)
            if resource in ['disco_ball']:  # Only preload images
                self.get_icon(resource)
        logger.info("Resource preloading complete") 